import abc
import torch as T
from utils.render_utils import get_sphere_intersections


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
	               SDF=None,
	               density_func=None):
		"""
		:param ray_dirs:  b x n x 3
		:param cam_loc: b x n x 3
		:param training: bool
		:return: z_vals b x n x n_s
		"""
		if training:
			N_samples = self.N_samples
		else:
			N_samples = self.N_samples_eval
		device = ray_dirs.get_device()
		b, n, d = ray_dirs.size()
		if not self.take_sphere_intersection:
			near = self.near * T.ones(b, n, 1).to(device)
			far = self.far * T.ones(b, n, 1).to(device)
		else:
			sphere_intersections = get_sphere_intersections(cam_loc,
			                                                ray_dirs,
			                                                r=self.scene_bounding_sphere)
			near = self.near * T.ones(b, n, 1).to(device)
			far = sphere_intersections[:, 1:]

		t_vals = T.linspace(0., 1., steps=N_samples).to(device)
		z_vals = near * (1. - t_vals) + far * (t_vals)

		if training:
			# get intervals between samples
			mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
			upper = T.cat([mids, z_vals[..., -1:]], -1)
			lower = T.cat([z_vals[..., :1], mids], -1)
			# stratified samples in those intervals
			t_rand = T.rand(z_vals.shape).to(device)

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
	               density_func=None):
		device = ray_dirs.get_device()
		beta0 = density_func.get_beta().detach()

		# Start with uniform sampling
		z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, SDF.training)
		samples = z_vals

		# Get maximum beta from the upper bound (Lemma 2)
		dists = z_vals[..., 1:] - z_vals[..., :-1]
		bound = (1.0 / (4.0 * T.log(T.tensor(self.eps + 1.0)))) * (dists ** 2.).sum(-1)
		beta = T.sqrt(bound)

		total_iters, not_converge = 0, True

		# Algorithm 1
		while not_converge and total_iters < self.max_total_iters:
			samples = z_vals
			points = cam_loc.unsqueeze(-2) + samples[..., None] * ray_dirs.unsqueeze(-2)
			points_flat = points.reshape(points.shape[0], -1, 3)

			# Calculating the SDF only for the new sampled points
			with T.no_grad():
				samples_sdf = SDF(pts=points_flat)
				sdf = samples_sdf

			# Calculating the bound d* (Theorem 1)
			d = sdf.reshape(z_vals.shape)
			dists = z_vals[..., 1:] - z_vals[..., :-1]
			a, b, c = dists, d[..., :-1].abs(), d[..., 1:].abs()
			first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
			second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
			d_star = T.zeros_like(z_vals)[:, :, :-1].to(device)
			d_star[first_cond] = b[first_cond]
			d_star[second_cond] = c[second_cond]
			s = (a + b + c) / 2.0
			area_before_sqrt = s * (s - a) * (s - b) * (s - c)
			mask = ~first_cond & ~second_cond & (b + c - a > 0)
			d_star[mask] = (2.0 * T.sqrt(area_before_sqrt[mask])) / (a[mask])
			d_star = (d[..., 1:].sign() * d[..., :-1].sign() == 1) * d_star  # Fixing the sign

			# Updating beta using line search
			curr_error = self.get_error_bound(beta0, density_func, sdf, z_vals, dists, d_star)
			beta[curr_error <= self.eps] = beta0
			beta_min, beta_max = beta0 * T.ones(z_vals.shape[0:2]).to(device) , beta
			for j in range(self.beta_iters):
				beta_mid = (beta_min + beta_max) / 2.
				curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), density_func, sdf, z_vals, dists, d_star)
				beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
				beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
			beta = beta_max

			# Upsample more points
			density = density_func(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))

			dists = T.cat([dists, T.tensor([1e10]).to(device).unsqueeze(0).repeat(dists.shape[0],
			                                                                      dists.shape[1], 1)], -1)
			free_energy = dists * density
			shifted_free_energy = T.cat([T.zeros(*dists.shape[0:2], 1).to(device),
			                             free_energy[..., :-1]], dim=-1)
			alpha = 1 - T.exp(-free_energy)
			transmittance = T.exp(-T.cumsum(shifted_free_energy, dim=-1))
			weights = alpha * transmittance  # probability of the ray hits something here

			#  Check if we are done and this is the last sampling
			total_iters += 1
			not_converge = beta.max() > beta0

			if not_converge and total_iters < self.max_total_iters:
				''' Sample more points proportional to the current error bound'''

				N = self.N_samples_eval

				bins = z_vals
				error_per_section = T.exp(-d_star / beta.unsqueeze(-1)) * (dists[..., :-1] ** 2.) / (4 * beta.unsqueeze(-1) ** 2)
				error_integral = T.cumsum(error_per_section, dim=-1)
				bound_opacity = (T.clamp(T.exp(error_integral), max=1.e6) - 1.0) * transmittance[..., :-1]

				pdf = bound_opacity + self.add_tiny
				pdf = pdf / T.sum(pdf, -1, keepdim=True)
				cdf = T.cumsum(pdf, -1)
				cdf = T.cat([T.zeros_like(cdf[..., :1]), cdf], -1)

			else:
				''' Sample the final sample set to be used in the volume utils integral '''

				N = self.N_samples

				bins = z_vals
				pdf = weights[..., :-1]
				pdf = pdf + 1e-5  # prevent nans
				pdf = pdf / T.sum(pdf, -1, keepdim=True)
				cdf = T.cumsum(pdf, -1)
				cdf = T.cat([T.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

			# Invert CDF
			if (not_converge and total_iters < self.max_total_iters) or (not SDF.training):
				u = T.linspace(0., 1., steps=N).to(device).unsqueeze(0).repeat(*cdf.shape[:-1], 1)
			else:
				u = T.rand(list(cdf.shape[:-1]) + [N]).to(device)
			u = u.contiguous()

			inds = T.searchsorted(cdf, u, right=True)
			below = T.max(T.zeros_like(inds - 1), inds - 1)
			above = T.min((cdf.shape[-1] - 1) * T.ones_like(inds), inds)
			inds_g = T.stack([below, above], -1)  # (batch, N_samples, 2)

			matched_shape = [inds_g.shape[0], inds_g.shape[1], inds_g.shape[2], cdf.shape[-1]]
			cdf_g = T.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
			bins_g = T.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)

			denom = (cdf_g[..., 1] - cdf_g[..., 0])
			denom = T.where(denom < 1e-5, T.ones_like(denom), denom)
			t = (u - cdf_g[..., 0]) / denom
			samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

			# Adding samples if we not converged
			if not_converge and total_iters < self.max_total_iters:
				z_vals, samples_idx = T.sort(T.cat([z_vals, samples], -1), -1)

		z_samples = samples

		near, far = self.near * T.ones(*ray_dirs.shape[:-1], 1).to(device), \
		            self.far * T.ones(*ray_dirs.shape[:-1], 1).to(device)
		if self.inverse_sphere_bg:  # if inverse sphere then need to add the far sphere intersection
			far = get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)[:, 1:]

		if self.N_samples_extra > 0:
			if density_func.training:
				sampling_idx = T.randperm(z_vals.shape[-1])[:self.N_samples_extra]
			else:
				sampling_idx = T.linspace(0, z_vals.shape[-1] - 1, self.N_samples_extra).long()
			z_vals_extra = T.cat([near, far, z_vals[..., sampling_idx]], -1)
		else:
			z_vals_extra = T.cat([near, far], -1)

		z_vals, _ = T.sort(T.cat([z_samples, z_vals_extra], -1), -1)

		if self.inverse_sphere_bg:
			z_vals_inverse_sphere = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, SDF.training)
			z_vals_inverse_sphere = z_vals_inverse_sphere * (1. / self.scene_bounding_sphere)
			z_vals = (z_vals, z_vals_inverse_sphere)

		return z_vals

	def get_error_bound(self, beta, density_func, sdf, z_vals, dists, d_star):
		density = density_func(sdf.reshape(z_vals.shape),beta=beta)
		b, n, d = dists.size()
		shifted_free_energy = T.cat([T.zeros(b, n, 1).to(dists.device),
		                             dists * density[..., :-1]], dim=-1)
		integral_estimation = T.cumsum(shifted_free_energy, dim=-1)
		error_per_section = T.exp(-d_star / beta) * (dists ** 2.) / (4 * beta ** 2)
		error_integral = T.cumsum(error_per_section, dim=-1)
		bound_opacity = (T.clamp(T.exp(error_integral), max=1.e6) - 1.0) * T.exp(-integral_estimation[..., :-1])

		return bound_opacity.max(-1)[0]


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
