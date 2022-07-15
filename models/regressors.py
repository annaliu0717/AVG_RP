import numpy as np
import torch
import torch.nn as nn
from models.points import Points
from models.sdf import SDF
from models.density import LaplaceDensity
from models.ray_sampler import ErrorBoundSampler, UniformSampler
from utils.render_utils import *


class ColorRegressor(nn.Module):
	def __init__(self, d_in, d_out, dims, weight_norm=True):
		super(ColorRegressor, self).__init__()
		self.d_in = d_in  # r,g,b,signed distance
		dims = [d_in] + dims + [d_out]
		self.num_layers = len(dims)
		self.activation = nn.ReLU()
		self.output_layer = nn.Sigmoid()

		self.model = []
		for i in range(self.num_layers - 1):
			layer = nn.Linear(dims[i], dims[i + 1])
			if weight_norm:
				layer = nn.utils.weight_norm(layer)
			if dims[i + 1] == d_out:
				self.model.append(layer)
			else:
				self.model.extend([layer, self.activation])
		self.model = nn.Sequential(*self.model)

	def forward(self,
	            # pts: T.tensor,
	            neighbors: T.tensor,
	            n_conf: T.tensor,
	            dist: T.tensor,  # b, n, k
	            # normals: T.tensor,
	            view_dirs: T.tensor,  # b, n, 3
	            # sd_vals: T.Tensor
	            ):

		# b, n, d = pts.shape
		# n_locs = neighbors[..., :3]
		# n_conf = neighbors[..., 3]  # b, n, k
		n_features = neighbors[..., 3:]  # b, n, k, f_dim
		w = 1 / dist
		w = (n_conf * (w / w.sum(dim=-1, keepdims=True))).unsqueeze(-1)  # b, n, k, 1
		f = (n_features * w).sum(dim=-2)  # b, n, f_dim
		# point_to_surface = (neighbors - pts[..., None, :]).sum(dim=-2)  # bn_cn_s, 3
		# assert neighbors.requires_grad
		# point_to_surface = F.normalize(point_to_surface, p=2)
		inputs = T.cat((view_dirs, f), -1)  # 3 + f_dim # 3 + 3 + 3xk + 3xk + 1
		return self.output_layer(self.model(inputs))


class RenderingNetwork(nn.Module):
	def __init__(self, conf, device, points=None, center=None):
		super(RenderingNetwork, self).__init__()
		self.device = device
		self.points = Points(points=points,
		                     center=center,
		                     device=self.device,
		                     **conf.get_config('points'))
		self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
		self.white_bkgd = conf.get_bool('white_bkgd', default=False)
		self.bg_color = T.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().to(self.device)
		self.sdf = SDF(conf.get_config('sdf'), self.points)
		self.density_func = LaplaceDensity(**conf.get_config('density'))
		self.color_network = ColorRegressor(**conf.get_config('color_regressor'))
		self.ray_sampler = UniformSampler(**conf.get_config('ray_sampler'))

	def rendering_func(self, density: T.tensor,
	                   color: T.tensor,
	                   z_vals: T.tensor) -> tuple:
		# print('density', density.max(), density.min())
		# print('z_vals', z_vals.max(), z_vals.min())
		dists = z_vals[..., 1:] - z_vals[..., :-1]  # bn_c x (n_s -1)
		dists = T.cat([dists,
		               T.tensor([1e10]).to(self.device).unsqueeze(0).repeat(*dists.shape[:-1], 1)], -1)
		weights = self.get_weights(density.view(*dists.size()), dists)
		rgb_map = T.sum(weights.unsqueeze(-1) * color, -2)
		if self.white_bkgd:
			acc_map = T.sum(weights, -1)
			rgb_map = rgb_map + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

		return rgb_map

	def get_weights(self, density: T.tensor, distance: T.tensor):
		"""
		:param density: b x n x 1 array
		:param distance: b x n x 1 array
		:return: b x n x 1 array
		"""
		fe = distance * density  # free_energy
		alpha = 1. - T.exp(-fe)
		shifted_fe = T.cat([T.zeros(*distance.shape[:-1], 1).to(device), fe[..., :-1]], dim=-1)
		t = T.exp(-T.cumsum(shifted_fe, dim=-1))
		weights = alpha * t
		return weights

	def forward(self, inputs: dict) -> T.tensor:
		"""
		:param pts: b x n x 3 locations of sampled points on the camera ray
		:param view_dirs: b x n x 3, viewing directions
		return: b x n x 1, density
		"""
		# Parse model input
		intrinsics = inputs["intrinsics"].to(self.device)
		uv = inputs["uv"].to(self.device)
		pose = inputs["pose"].to(self.device)

		# ray_dirs: b x n x 3
		# camera_loc: b x 3
		ray_dirs, cam_loc_ori = get_camera_params(uv, pose, intrinsics)
		batch_size, num_pixels, _ = ray_dirs.shape

		cam_loc = cam_loc_ori.unsqueeze(1).repeat(1, num_pixels, 1)  # b x 1 x 3 -> b x n x 3 -> (bn ,3)
		ray_dirs = ray_dirs.reshape(batch_size, -1, 3)  # bn x 3

		self.points.update_uv(intrinsics, pose)
		z_vals = self.ray_sampler.get_z_vals(ray_dirs,
		                                     cam_loc,
		                                     training=self.sdf.training,
		                                     SDF=self.sdf,
		                                     density_func=self.density_func)
		N_samples = z_vals.shape[-1]  # bn x n_s

		pts = cam_loc.unsqueeze(-2) + z_vals.unsqueeze(-1) * ray_dirs.unsqueeze(-2)  # bn x 1 x 3 + bn x ns x 1 * bn x 1 x 3
		pts = pts.reshape(batch_size, -1, 3)  # in world coordinate b*n_cam x n_sample x 3

		dirs = ray_dirs.unsqueeze(-2).repeat(1, 1, N_samples, 1)
		dirs_flat = dirs.reshape(batch_size, -1, 3)

		sd_vals = self.sdf(pts)  # b, n_c * n_s , 1
		import matplotlib.pyplot as plt
		plot = 0
		if plot:
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')
			ax.scatter(self.points.get_points_loc()[::50, 0].detach().cpu().numpy(),
			           self.points.get_points_loc()[::50, 1].detach().cpu().numpy(),
			           self.points.get_points_loc()[::50, 2].detach().cpu().numpy(),
			           s=1, c='m')
			ax.scatter(pts[0, ::10, 0].detach().cpu().numpy(),
			           pts[0, ::10, 1].detach().cpu().numpy(),
			           pts[0, ::10, 2].detach().cpu().numpy(),
			           c=T.sign(sd_vals[0, ::10]).detach().cpu().numpy(),
			           s=1)
			plt.show()
			plt.close(fig)
		#
		# def normal_sampler(cam_loc, rays, N_samples):
		# 	vec = F.normalize(rays, dim=-1)  # b x n x 3
		# 	cam_loc = cam_loc.unsqueeze(1).repeat(1, rays.shape[1], 1)  # b x n x 3
		# 	t_vals = T.sort(
		# 		((T.fmod(T.randn(size=(*rays.shape[:2], N_samples)), 1) * 0.2) + 1.).to(rays.device)).values
		# 	pts_near_surface = rays.unsqueeze(-2) * t_vals.unsqueeze(-1)
		# 	z_vals = (pts_near_surface / vec.unsqueeze(-2))[..., 0]
		# 	pts_near_surface = pts_near_surface + cam_loc.unsqueeze(-2)
		# 	# pts_near_surface = cam_loc.unsqueeze(-2) + (z_vals.unsqueeze(-1) * vec.unsqueeze(-2))
		# 	assert pts_near_surface.shape[-1] == 3
		# 	# print(z_vals.max())
		# 	return pts_near_surface, z_vals

		# nn_uvs = self.points.get_uvs(self.sdf.get_nn_idx())
		density = self.density_func(sd_vals)  # b*n_c * n_s , 1
		# sample points around surface
		color = self.color_network(self.sdf.nn_points,
		                           self.sdf.nn_conf,
		                           self.sdf.nn_dist,
		                           dirs_flat).view(batch_size,
		                                           num_pixels,
		                                           N_samples,
		                                           3)  # b*n_c*n_s, 3
		# bn_c x n_s
		rgb_map = self.rendering_func(density.reshape(batch_size,
		                                              num_pixels,
		                                              N_samples,
		                                              1),
		                              color,
		                              z_vals)
		if self.training:
			output = {
				'rgb_map': rgb_map,  # b x n x 3
				'confs': self.sdf.nn_conf
			}
		# del sd_vals_near_surface, pts_near_surface, z_vals_surface, sd_vals, pts
		else:
			output = {
				'rgb_map': rgb_map,  # b x n x 3
				# 'rgb_near_surface': rgb_map_near_surface,
				'sd_vals': sd_vals[0, ::N_samples * 100],  # b x (n_sample x n_cam)
				'samples': pts[0, ::N_samples * 100]  # b x n_cam x n_sample x 3
			}
			del sd_vals, pts

		return output
