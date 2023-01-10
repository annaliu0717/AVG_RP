import abc
import torch as torch
from utils.rend_utils import get_sphere_intersections
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import pytorch3d.ops as ops


def sample_pdf(bins, weights, N_samples, det=False):
	# Get pdf
	pdf = weights[..., :-1]
	pdf = pdf + 1e-5  # prevent nans
	pdf = pdf / torch.sum(pdf, -1, keepdim=True)
	cdf = torch.cumsum(pdf, -1)
	cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

	# Take uniform samples
	if det:
		u = torch.linspace(0., 1., steps=N_samples)
		u = u.expand(list(cdf.shape[:-1]) + [N_samples]).to(cdf.device)
	else:
		u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

	# Invert CDF
	u = u.contiguous().to(cdf.device)
	inds = torch.searchsorted(cdf, u, right=True).to(cdf.device)
	below = torch.max(torch.zeros_like(inds - 1), inds - 1)
	above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
	inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

	# cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
	# bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
	matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]]
	cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
	bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)

	denom = (cdf_g[..., 1] - cdf_g[..., 0])
	denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
	t = (u - cdf_g[..., 0]) / denom
	samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

	return samples


class RaySampler(metaclass=abc.ABCMeta):
	def __init__(self, near, far):
		self.near = near
		self.far = far

	@abc.abstractmethod
	def get_z_vals(self, ray_dirs, cam_loc):
		pass


class UniformSampler(RaySampler):
	def __init__(self, scene_bounding_sphere, near, N_samples, N_samples_eval,
	             take_sphere_intersection=False, far: float = -1.0):
		super().__init__(near, 2.0 * scene_bounding_sphere if far == -1.0 else far)  # default far is 2*R
		self.N_samples = N_samples
		self.N_samples_eval = N_samples_eval
		self.scene_bounding_sphere = scene_bounding_sphere
		self.take_sphere_intersection = take_sphere_intersection

	def get_z_vals(self, ray_dirs,
	               cam_loc,
	               training: bool = True,
	               N_samples = None):
		"""
		:param ray_dirs:  b x n x 3
		:param cam_loc: b x n x 3
		:param training: bool
		:return: z_vals b x n x n_s
		"""
		if N_samples is None:
			if training:
				N_samples = self.N_samples
			else:
				N_samples = self.N_samples_eval
		device = ray_dirs.get_device()
		#b, n, d = ray_dirs.size()
		if not self.take_sphere_intersection:
			near = self.near * torch.ones(*ray_dirs.shape[:-1], 1).to(device)
			far = self.far * torch.ones(*ray_dirs.shape[:-1], 1).to(device)
		else:
			sphere_intersections = get_sphere_intersections(cam_loc,
			                                                ray_dirs,
			                                                r=self.scene_bounding_sphere)
			near = self.near * torch.ones(*ray_dirs.shape[:-1], 1).to(device)
			far = sphere_intersections[:, 1:]

		t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
		z_vals = near * (1. - t_vals) + far * (t_vals)

		if training:
			# get intervals between samples
			mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
			upper = torch.cat([mids, z_vals[..., -1:]], -1)
			lower = torch.cat([z_vals[..., :1], mids], -1)
			# stratified samples in those intervals
			t_rand = torch.rand(z_vals.shape).to(device)

			z_vals = lower + (upper - lower) * t_rand

		return z_vals


class ErrorBoundSampler(RaySampler):
	def __init__(self, scene_bounding_sphere, near, N_samples, N_samples_eval, N_samples_extra,
	             eps, beta_iters, max_total_iters,
	             inverse_sphere_bg=False, N_samples_inverse_sphere=0, add_tiny=0.0):
		super().__init__(near, 2.0 * scene_bounding_sphere)
		self.N_samples = N_samples
		self.N_samples_eval = N_samples_eval
		self.uniform_sampler = UniformSampler(scene_bounding_sphere,
		                                      near,
		                                      N_samples_eval,
		                                      N_samples_eval,
		                                      take_sphere_intersection=inverse_sphere_bg)

		self.N_samples_extra = N_samples_extra

		self.eps = eps
		self.beta_iters = beta_iters
		self.max_total_iters = max_total_iters
		self.scene_bounding_sphere = scene_bounding_sphere
		self.add_tiny = add_tiny

		self.inverse_sphere_bg = inverse_sphere_bg
		if inverse_sphere_bg:
			self.inverse_sphere_sampler = UniformSampler(1.0, 0.0, N_samples_inverse_sphere, False, far=1.0)

	def get_z_vals(self,
	               ray_dirs,
	               cam_loc,
	               training=True,
	               SDF=None,
	               density_func=None,
	               z_init=None):

		torch.cuda.empty_cache()
		device = ray_dirs.get_device()
		beta0 = density_func.get_beta().detach()

		# Start with uniform sampling
		if z_init is None:
			z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, SDF.training)
		else:
			z_vals, _ = torch.sort(z_init, dim=-1)

		samples, samples_idx = z_vals, None

		# Get maximum beta from the upper bound (Lemma 2)
		dists = z_vals[..., 1:] - z_vals[..., :-1]
		bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists ** 2.).sum(-1)
		beta = torch.sqrt(bound)

		total_iters, not_converge = 0, True


		# Algorithm 1
		while not_converge and total_iters < self.max_total_iters:
			points = cam_loc.unsqueeze(-2) + samples[..., None] * ray_dirs.unsqueeze(-2)
			# points_flat = points.reshape(points.shape[0], -1, 3)

			# Calculating the SDF only for the new sampled points
			# Calculating the SDF only for the new sampled points
			with torch.no_grad():
				samples_sdf, _ = SDF(pts=points.reshape(points.shape[0], -1, 3))
			if samples_idx is not None:
				sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
				                       samples_sdf.reshape(-1, samples.shape[1])], -1)
				sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
			else:
				sdf = samples_sdf


			# Calculating the bound d* (Theorem 1)
			d = sdf.reshape(z_vals.shape)
			dists = z_vals[..., 1:] - z_vals[..., :-1]
			a, b, c = dists, d[..., :-1].abs(), d[..., 1:].abs()
			first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
			second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
			d_star = torch.zeros_like(z_vals)[..., :-1].to(device)
			d_star[first_cond] = b[first_cond]
			d_star[second_cond] = c[second_cond]
			s = (a + b + c) / 2.0
			area_before_sqrt = s * (s - a) * (s - b) * (s - c)
			mask = ~first_cond & ~second_cond & (b + c - a > 0)
			d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
			d_star = (d[..., 1:].sign() * d[..., :-1].sign() == 1) * d_star  # Fixing the sign

			# Updating beta using line search
			curr_error = self.get_error_bound(beta0, density_func, sdf, z_vals, dists, d_star)
			beta[curr_error <= self.eps] = beta0
			beta_min, beta_max = beta0 * torch.ones(z_vals.shape[:-1]).to(device), beta
			for j in range(self.beta_iters):
				beta_mid = (beta_min + beta_max) / 2.
				curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), density_func, sdf, z_vals, dists, d_star)
				beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
				beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
			beta = beta_max

			# Upsample more points
			density = density_func(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))

			dists = torch.cat([dists, torch.tensor([1e10]).to(device).unsqueeze(0).repeat(*dists.shape[:-1], 1)], -1)
			free_energy = dists * density
			shifted_free_energy = torch.cat([torch.zeros(*dists.shape[:-1], 1).to(device),
			                                 free_energy[..., :-1]], dim=-1)
			alpha = 1 - torch.exp(-free_energy)
			transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
			weights = alpha * transmittance  # probability of the ray hits something here
			del free_energy
			#  Check if we are done and this is the last sampling
			total_iters += 1
			not_converge = beta.max() > beta0

			if not_converge and total_iters < self.max_total_iters:
				''' Sample more points proportional to the current error bound'''
				N = self.N_samples_eval

				bins = z_vals
				error_per_section = torch.exp(-d_star / (beta.unsqueeze(-1))) * (dists[..., :-1] ** 2.) / (4 * beta.unsqueeze(-1) ** 2)
				error_integral = torch.cumsum(error_per_section, dim=-1)
				bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * transmittance[..., :-1]

				pdf = bound_opacity + self.add_tiny
				pdf = pdf / torch.sum(pdf, -1, keepdim=True)
				cdf = torch.cumsum(pdf, -1)
				cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

			else:
				''' Sample the final sample set to be used in the volume utils integral '''

				N = self.N_samples# if training else self.N_samples_eval

				bins = z_vals
				pdf = weights[..., :-1]
				pdf = pdf + 1e-5  # prevent nans
				pdf = pdf / torch.sum(pdf, -1, keepdim=True)
				cdf = torch.cumsum(pdf, -1)
				cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

			# Invert CDF
			if (not_converge and total_iters < self.max_total_iters) or (not training):
				u = torch.linspace(0., 1., steps=N).to(device).unsqueeze(0).repeat(*cdf.shape[:-1], 1)
			else:
				u = torch.rand(list(cdf.shape[:-1]) + [N]).to(device)
			u = u.contiguous()

			inds = torch.searchsorted(cdf, u, right=True)
			below = torch.max(torch.zeros_like(inds - 1), inds - 1)
			above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
			inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

			matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]]
			cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
			bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)

			denom = (cdf_g[..., 1] - cdf_g[..., 0])
			denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
			t = (u - cdf_g[..., 0]) / denom
			samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

			# Adding samples if we not converged
			if not_converge and total_iters < self.max_total_iters:
				z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)

		z_samples = samples

		near, far = self.near * torch.ones(*ray_dirs.shape[:-1], 1).to(device), \
		            self.far * torch.ones(*ray_dirs.shape[:-1], 1).to(device)
		if self.inverse_sphere_bg:  # if inverse sphere then need to add the far sphere intersection
			far = get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)[:, 1:]

		#N_samples_extra = self.N_samples_extra if training else 0
		if self.N_samples_extra > 0:
			if density_func.training:
				sampling_idx = torch.randperm(z_vals.shape[-1])[:self.N_samples_extra]
			else:
				sampling_idx = torch.linspace(0, z_vals.shape[-1] - 1, self.N_samples_extra).long()
			z_vals_extra = torch.cat([near, far, z_vals[..., sampling_idx]], -1)
		else:
			z_vals_extra = torch.cat([near, far], -1)

		z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

		if self.inverse_sphere_bg:
			z_vals_inverse_sphere = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, SDF.training)
			z_vals_inverse_sphere = z_vals_inverse_sphere * (1. / self.scene_bounding_sphere)
			z_vals = (z_vals, z_vals_inverse_sphere)

		return z_vals

	def get_error_bound(self, beta, density_func, sdf, z_vals, dists, d_star):
		density = density_func(sdf.reshape(z_vals.shape), beta=beta)
		shifted_free_energy = torch.cat([torch.zeros(*dists.shape[:-1], 1).to(dists.device),
		                                 dists * density[..., :-1]], dim=-1)
		integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
		error_per_section = torch.exp(-d_star / (beta.abs())) * (dists ** 2.) / (4 * beta ** 2 + 1e-5)
		error_integral = torch.cumsum(error_per_section, dim=-1)
		bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * torch.exp(-integral_estimation[..., :-1])

		return bound_opacity.max(-1)[0]

from models.mesh import Mesh
class NearSurfaceSampler(RaySampler):
	def __init__(self, scene_bounding_sphere, near, N_samples, N_samples_eval, N_samples_extra,
	             eps, beta_iters, max_total_iters,
	             inverse_sphere_bg=False, N_samples_inverse_sphere=0, add_tiny=0.0,
	             ):
		super().__init__(near, 2.0 * scene_bounding_sphere)
		self.scene_bounding_sphere = scene_bounding_sphere
		self.N_samples = N_samples
		self.N_samples_eval = N_samples_eval
		self.N_samples_extra = N_samples_extra
		self.max_total_iters = max_total_iters
		self.obj_sampler = ErrorBoundSampler(scene_bounding_sphere, near, N_samples, N_samples_eval, N_samples_extra,
		                                     eps, beta_iters, max_total_iters,
		                                     inverse_sphere_bg, N_samples_inverse_sphere, add_tiny)
		self.uniform_sampler = UniformSampler(near=near,
		                                      scene_bounding_sphere=scene_bounding_sphere,
		                                      N_samples=N_samples,
		                                      N_samples_eval=N_samples_eval
		                                     )

	def get_z_vals(self, ray_dirs, cam_loc, training=False,
	               surface=None, SDF=None, density_func=None):
		device = surface.device
		N_samples = self.N_samples
		b, n_r, d = ray_dirs.size()

		cam_pc_rays = surface.get_cam_to_vert_rays(cam_loc)# b n_pc 3
		pixel_rays = F.normalize(ray_dirs, dim=-1).float()  # b n_pixel 3
		mu = (pixel_rays.unsqueeze(-2) * cam_pc_rays.unsqueeze(1)).sum(dim=-1) # b n_pixel n_points 3

		min_dists = (cam_pc_rays.unsqueeze(1) - (pixel_rays.unsqueeze(-2) * mu[..., None])).norm(dim=-1)
		k=15
		min_dists, sorted_id = min_dists.topk(k, dim=-1, largest=False, sorted=True)
		mu = mu.gather(-1, sorted_id) # b x n x k
		mu, _ = mu.sort(dim=-1)

		n_splite=4
		t = torch.linspace(0, 1, n_splite).view(1, 1, -1).to(device)
		mu = (mu[..., 1:, None]) * t + (mu[..., :-1, None] ) * (1 - t)
		mu = mu.view(*ray_dirs.shape[:-1], -1)
		mu, _ = mu.sort(dim=-1)

		points = cam_loc[..., None, :] + mu[..., None] * pixel_rays[..., None, :]
		with torch.no_grad():
			sdf, _ = SDF(pts=points.reshape(points.shape[0], -1, 3))

		density = density_func(sdf.reshape(mu.shape))
		dists = mu[..., 1:] - mu[..., :-1]
		dists = torch.cat([dists, torch.tensor([1e10]).to(device).unsqueeze(0).repeat(*dists.shape[:-1], 1)], -1)
		free_energy = dists * density
		shifted_free_energy = torch.cat([torch.zeros(*dists.shape[:-1], 1).to(device), free_energy[..., :-1]], dim=-1)
		alpha = 1 - torch.exp(-free_energy)
		transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
		weights = alpha * transmittance
		z_vals_near_surf = sample_pdf(mu, weights, N_samples, det=True)
		z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, training=training)
		z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_near_surf], -1), -1)

		z_vals = self.obj_sampler.get_z_vals(ray_dirs,
			                                   cam_loc,
			                                   training=training,
			                                   SDF=SDF,
			                                   density_func=density_func,
			                                   z_init=z_vals.detach()
			                                   )

		return z_vals



class DepthSampler(RaySampler):
	def __init__(self, scene_bounding_sphere, near, N_samples, N_samples_eval, N_samples_extra,
	             eps, beta_iters, max_total_iters,
	             inverse_sphere_bg=False, N_samples_inverse_sphere=0, add_tiny=0.0, upsample=2,
	             ):
		super().__init__(near, 2.0 * scene_bounding_sphere)
		self.scene_bounding_sphere = scene_bounding_sphere
		self.N_samples = N_samples
		self.N_samples_eval = N_samples_eval
		self.N_samples_extra = N_samples_extra
		self.upsample = max_total_iters
		self.training_sampler = NearSurfaceSampler(scene_bounding_sphere, near, N_samples, 2*N_samples, N_samples_extra*2,
		                                           eps, beta_iters, max_total_iters,
		                                           inverse_sphere_bg, N_samples_inverse_sphere, add_tiny)
		# self.training_sampler = ErrorBoundSampler(scene_bounding_sphere, near, N_samples, 2 * N_samples,
		#                                       N_samples_extra, eps, beta_iters, max_total_iters,
		#                                       inverse_sphere_bg, N_samples_inverse_sphere, add_tiny)
		self.uniform_sampler = UniformSampler(near=near,
		                                      scene_bounding_sphere=scene_bounding_sphere,
		                                      N_samples=N_samples,
		                                      N_samples_eval=N_samples_eval
		                                      )


	def get_z_vals(self, ray_dirs, cam_loc, surface=None, r=0.05,
	               SDF=None, D=None, training=True):
		device = ray_dirs.device
		b, n_r, d = ray_dirs.size()
		N_samples = training * self.N_samples + (1 - training) * self.N_samples_eval
		if (not training) :
			z_vals = self.init_sample(surface, r, N_samples, device)
			return z_vals

		# bg_idx = surface.squeeze(-1) < 0
		# bg_rays = ray_dirs[bg_idx, :]
		# cam_bg = cam_loc[bg_idx, :]
		#
		# z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, True)
		#
		# N = z_vals.shape[-1]
		# near = torch.clamp(surface - r, min=self.near)
		# far = torch.clamp(surface + r, max=self.far)
		#
		#
		# t = torch.linspace(0, 1, self.N_samples_extra).to(device)
		#
		# z_vals_obj = far * t + near * (1 - t)
		#
		# if training:
		# 	# get intervals between samples
		# 	mids = .5 * (z_vals_obj[..., 1:] + z_vals_obj[..., :-1])
		# 	upper = torch.cat([mids, z_vals_obj[..., -1:]], -1)
		# 	lower = torch.cat([z_vals_obj[..., :1], mids], -1)
		# 	# stratified samples in those intervals
		# 	t_rand = torch.rand(z_vals_obj.shape).to(device)
		#
		# 	z_vals_obj = lower + (upper - lower) * t_rand
		#
		# z_vals = torch.cat([z_vals, z_vals_obj], -1)
		#
		# if bg_rays.shape[0] > 0:
		# 	z_vals_bg = z_vals[bg_idx, :]
		# 	z_vals_bg = z_vals_bg[..., :N]
		# 	points = cam_bg[..., None, :] + z_vals_bg[..., None] * bg_rays[..., None, :]
		# 	with torch.no_grad():
		# 		sdf, _ = SDF(pts=points.reshape(-1, 3))
		# 	density = D(sdf.reshape(z_vals_bg.shape))
		# 	dists = z_vals_bg[..., 1:] - z_vals_bg[..., :-1]
		# 	dists = torch.cat([dists, torch.tensor([1e10]).to(device).unsqueeze(0).repeat(*dists.shape[:-1], 1)], -1)
		# 	free_energy = dists * density
		# 	shifted_free_energy = torch.cat([torch.zeros(*dists.shape[:-1], 1).to(device), free_energy[..., :-1]], dim=-1)
		# 	alpha = 1 - torch.exp(-free_energy)
		# 	transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
		# 	weights = alpha * transmittance
		# 	z_vals_mid = .5 * (z_vals_bg[..., 1:] + z_vals_bg[..., :-1])
		# 	z_samples = sample_pdf(z_vals_mid, weights[..., 1:], self.N_samples_extra, det=not (training))
		# 	z_vals_bg, _ = torch.sort(torch.cat([z_vals_bg, z_samples], -1), -1)
		# 	z_vals[bg_idx, :] = z_vals_bg
		#
		# z_vals, _ = torch.sort(z_vals, -1)
		#
		# self.training_sampler.max_total_iters = self.upsample

		z_vals = self.training_sampler.get_z_vals(ray_dirs,
		                                          cam_loc,
		                                          training=training,
		                                          SDF=SDF,
		                                          # z_init=z_vals.detach(),
		                                          surface=surface,
		                                          density_func = D)

		# for us in range(self.upsample):
		# 	n = int(0.5 * self.N_samples_extra)
		# 	points = cam_loc[..., None, :] + z_vals[..., None] * ray_dirs[..., None, :]
		# 	with torch.no_grad():
		# 		sdf, _ = SDF(pts=points.reshape(-1, 3))
		# 	density = D(sdf.reshape(z_vals.shape))
		# 	dists = z_vals[..., 1:] - z_vals[..., :-1]
		# 	dists = torch.cat([dists, torch.tensor([1e10]).to(device).unsqueeze(0).repeat(*dists.shape[:-1], 1)], -1)
		# 	free_energy = dists * density
		# 	shifted_free_energy = torch.cat([torch.zeros(*dists.shape[:-1], 1).to(device), free_energy[..., :-1]], dim=-1)
		# 	alpha = 1 - torch.exp(-free_energy)
		# 	transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
		# 	weights = alpha * transmittance
		# 	z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
		# 	z_samples = sample_pdf(z_vals_mid, weights[..., 1:], n, det=not(training))
		# 	z_samples = z_samples.detach()
		# 	z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

		# if not uniform_init:
		# 	near, far = self.near * torch.ones(*ray_dirs.shape[:-1], 1).to(device), \
		# 	            self.far * torch.ones(*ray_dirs.shape[:-1], 1).to(device)
		# 	z_vals_extra = torch.cat([near, far], -1)
		# 	z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_extra], -1), -1)

		# print(((surface > 0) & (near_bounds < 0)).sum())
		return z_vals

	def init_sample(self, surface, r, N_samples, device):
		near = torch.clamp(surface - r, min=self.near)
		far = torch.clamp(surface + r, max=self.far)
		# sphere_intersections = get_sphere_intersections(cam_loc,
		#                                                 ray_dirs,
		#                                                 r=self.scene_bounding_sphere)
		#a = sphere_intersections[surface.squeeze(-1)<0, ...]
		near[surface < 0] = self.near
		far[surface < 0] = self.far

		t = torch.linspace(0, 1, N_samples).to(device)
		z_vals = far * t + near * (1 - t)
		# self.eval_sampler.max_total_iters = upsample
		# z_vals = self.eval_sampler.get_z_vals(ray_dirs,
		#                                       cam_loc,
		#                                       training=not(eval),
		#                                       SDF=SDF,
		#                                       z_init=z_vals.detach(),
		#                                       density_func=D)

		# if self.N_samples_extra > 0:
		# 	for u in range(upsample):
		# 		points = cam_loc[..., None, :] + z_vals[..., None] * ray_dirs[..., None, :]
		# 		with torch.no_grad():
		# 			sdf, _ = SDF(pts=points.reshape(points.shape[0], -1, 3))
		# 		density = D(sdf.reshape(z_vals.shape))
		# 		dists = z_vals[..., 1:] - z_vals[..., :-1]
		# 		dists = torch.cat([dists, torch.tensor([1e10]).to(device).unsqueeze(0).repeat(*dists.shape[:-1], 1)], -1)
		# 		free_energy = dists * density
		# 		shifted_free_energy = torch.cat([torch.zeros(*dists.shape[:-1], 1).to(device), free_energy[..., :-1]], dim=-1)
		# 		alpha = 1 - torch.exp(-free_energy)
		# 		transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
		# 		weights = alpha * transmittance
		# 		z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
		# 		z_samples = sample_pdf(z_vals_mid, weights[..., 1:], N_samples, det=eval)
		# 		z_samples = z_samples.detach()
		# 		z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

		# near, far = self.near * torch.ones(*ray_dirs.shape[:-1], 1).to(device), \
		#             self.far * torch.ones(*ray_dirs.shape[:-1], 1).to(device)
		# z_vals_extra = torch.cat([near, far], -1)
		#
		# z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_extra], -1), -1)

		return z_vals

def main():
	conf = {
		"scene_bounding_sphere": 3.0,
		'near': 0.0,
		"N_samples": 64,
		"N_samples_eval": 128,
		"N_samples_extra": 32,
		"eps": 0.1,
		"beta_iters": 10,
		"max_total_iters": 5,
		"N_samples_inverse_sphere": 32,
		"add_tiny": 1.0e-6
	}
	ray_sampler = ErrorBoundSampler(**conf)


if __name__ == "__main__":
	main()
