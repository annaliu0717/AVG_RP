import torch as T
import torch.nn as nn
from models.points import sphere_generator
from models.regressors import Regressor
from models.ray_sampler import *
from utils.render_utils import *

class Renderer(nn.Module):
	def __init__(self, ray_batch, color_batch, **kwargs):
		super().__init__()
		n_points = kwargs.get("n_points", 10000)
		self.device = kwargs.get("device", 'cpu')
		self.n_samples = kwargs.get("n_samples", 100)
		self.feature_vector_size = kwargs.get('feature_vector_size')
		self.scene_bounding_sphere = kwargs.get('scene_bounding_sphere', default=1.0)
		self.white_bkgd = kwargs.get('white_bkgd', False)
		self.bg_color = T.tensor(kwargs.get("bg_color", [1.0, 1.0, 1.0])).float().to(self.device)

		points =nn.Parameter(sphere_generator(n_points))
		k = kwargs.get("k", 8)
		self.n_importance = kwargs.get("N_importance", 0)
		self.pytest = kwargs.get("pytest", False)
		self.perturb = kwargs.get("perturb", 0)
		self.regressor = Regressor(points, k, self.device, kwargs)
		self.sampler = UniformSampler()

	def get_alpha(density, distance):
		return 1 - T.exp(- density * distance)

	def get_weights(self, density: T.tensor, distance: T.tensor):
		"""
		:param density: b x n x 1 array
		:param distance: b x n x 1 array
		:return: b x n x 1 array
		"""
		alpha = self.get_alpha(density, distance)
		weights = alpha * T.cumprod(T.cat([T.ones((alpha.size()[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
		return weights

	def rendering_func(self, density: T.tensor,
	                   color: T.tenso,
	                   distance: T.tenso,
	                   z_vals: T.tenso) -> tuple:
		weights = self.get_weights(density, distance)
		rgb_map = T.sum(weights[..., None] * color, -2)
		depth_map = T.sum(weights * z_vals, -1)
		disp_map = 1. / T.max(1e-10 * T.ones_like(depth_map), depth_map / T.sum(weights, -1))
		acc_map = T.sum(weights, -1)
		return rgb_map, depth_map, disp_map, acc_map

	def forward(self, ray_batch: T.tensor, color_batch: T.tensor):
		n_rays, d = ray_batch.size()
		rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
		# Extract unit-normalized viewing direction.
		viewdirs = ray_batch[:, -3:] if d > 8 else None
		# Extract lower, upper bound for ray distance.
		bounds = ray_batch[..., 6:8].view(-1, 1, 2)
		near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

		t_vals = T.linspace(0, 1, self.n_samples)
		z_vals = near * (1 - t_vals) + far * (t_vals)
		z_vals = z_vals.expand(n_rays, self.n_samples)
		dists = z_vals[..., 1:] - z_vals[..., :-1]
		dists = T.cat([dists, T.Tensor([1e10]).expand(*dists[..., :1].size())], dim=-1)
		pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
		density, color = self.regressor(pts, color_batch)
		rgb_map, disp_map, acc_map, weights, depth_map =  self.rendering_func(density,
		                                                                      color,
		                                                                      dists,
		                                                                      z_vals)
		ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
		if self.n_importance > 0:
			rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

			z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
			z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1],
			                       self.n_importance,
			                       det=(self.perturb == 0.),
			                       pytest=self.pytest)
			z_samples = z_samples.detach()
			z_vals, _ = T.sort(T.cat([z_vals, z_samples], -1), -1)
			pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
			density, color = self.regressor(pts, color_batch)
			rgb_map, disp_map, acc_map, weights, depth_map = self.rendering_func(density,
			                                                                     color,
			                                                                     dists,
			                                                                     z_vals)

		return ret

