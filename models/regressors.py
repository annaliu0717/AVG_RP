import numpy as np
import torch
import torch.nn as nn
from models.points import Points
from models.sdf import SDF
from models.density import LaplaceDensity
from models.ray_sampler import *
from utils.rend_utils import *
from models.embedder import get_embedder
from models.mesh import Mesh


class ColorRegressor(nn.Module):
	def __init__(
					self,
					mode,
					d_in,
					d_out,
					dims,
					weight_norm=True,
					num_diff_layers=3,
					multires=0,
					multires_view=0,
	):
		super().__init__()

		self.mode = mode
		self.num_diff_layers = num_diff_layers
		dims_diff = [d_in] + dims[:self.num_diff_layers]
		dims_specular = [dims[self.num_diff_layers - 1] + 3 + 3 + 3] + dims[(self.num_diff_layers):] + [d_out]
		self.num_specular_layers = len(dims_specular)
		# dims = [d_in + feature_vector_size] + dims + [d_out]

		self.embed_fn = None
		self.embedview_fn = None
		if multires > 0:
			embed_fn, input_ch = get_embedder(multires)
			self.embed_fn = embed_fn
			#dims[0] += (input_ch - 3)
			dims_diff[0] += (input_ch - 3)
		if multires_view > 0:
			embedview_fn, input_ch = get_embedder(multires_view)
			self.embedview_fn = embedview_fn # dims[0] += (input_ch - 3)
			dims_specular[0] += (input_ch - 3)

		self.num_layers = len(dims)

		for l in range(0, self.num_diff_layers - 1):
			out_dim_diff = dims_diff[l + 1]
			lin_diff = nn.Linear(dims_diff[l], out_dim_diff)

			if weight_norm:
				lin_diff = nn.utils.weight_norm(lin_diff)
			setattr(self, "lin_diff" + str(l), lin_diff)

		for l in range(0, self.num_specular_layers - 1):
			out_dims_specular = dims_specular[l + 1]
			lin_dims_specular = nn.Linear(dims_specular[l], out_dims_specular)

			if weight_norm:
				lin_dims_specular = nn.utils.weight_norm(lin_dims_specular)
			setattr(self, "lin_spec" + str(l), lin_dims_specular)


		#for l in range(0, self.num_layers - 1):
		#	out_dim = dims[l + 1]
		#	lin = nn.Linear(dims[l], out_dim)

		#	if weight_norm:
		#		lin = nn.utils.weight_norm(lin)

		#	setattr(self, "lin" + str(l), lin)

		self.relu = nn.ReLU()
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, points, view_dirs, normals, feature_vectors=None, position=None):
		if position is None:
			position = points
		if self.embed_fn is not None:
			points = self.embed_fn(points)
		if self.embedview_fn is not None:
			view_dirs = self.embedview_fn(view_dirs)

		if self.mode == 'idr':
			rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
		elif self.mode == 'nerf':
			rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
		elif self.mode == 'naive':
			rendering_input = torch.cat([points, view_dirs, normals], dim=-1)
		elif self.mode == 'nds':
			rendering_input = points

		x = rendering_input

		for l in range(0, self.num_diff_layers - 1):
			lin = getattr(self, "lin_diff" + str(l))
			x = lin(x)
			x = self.relu(x)

		x = torch.cat([x, position, view_dirs, normals], dim=-1)

		for l in range(0, self.num_specular_layers - 1):
			lin = getattr(self, "lin_spec" + str(l))
			x = lin(x)
			if l < self.num_specular_layers - 2:
				x = self.relu(x)

		x = self.sigmoid(x)
		return x

		# for l in range(0, self.num_layers - 1):
		# 	lin = getattr(self, "lin" + str(l))
		#
		# 	x = lin(x)
		#
		# 	if l < self.num_layers - 2:
		# 		x = self.relu(x)
		#
		# x = self.sigmoid(x)
		# return x


class RenderingNetwork(nn.Module):
	def __init__(self, conf, device, verts=None, faces=None, vert_normals=None, center=None):
		super(RenderingNetwork, self).__init__()
		self.device = device
		mesh = Mesh(vertices=verts,
		            faces=faces,
		            device=self.device)

		self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
		self.white_bkgd = conf.get_bool('white_bkgd', default=False)
		self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().to(self.device)

		self.density_func = LaplaceDensity(**conf.get_config('density'))
		self.sdf = SDF(**conf.get_config('sdf'), mesh=mesh)
		self.color_network = ColorRegressor(**conf.get_config('color_regressor'))
		self.ray_sampler = DepthSampler(**conf.get_config('ray_sampler'))

	def rendering_func(self, density: torch.tensor,
	                   color: torch.tensor,
	                   z_vals: torch.tensor) -> tuple:

		dists = z_vals[..., 1:] - z_vals[..., :-1]  # bn_c x (n_s -1)
		dists = torch.clamp(dists, min=1e-8)
		dists = torch.cat([dists,
		                   torch.tensor([1e10]).to(self.device).unsqueeze(0).repeat(*dists.shape[:-1], 1)], -1)
		weights = self.get_weights(density.view(*dists.size()), dists)
		depth_map = torch.sum(weights * z_vals, -1)
		rgb_map = torch.sum(weights.unsqueeze(-1) * color, -2)
		if self.white_bkgd:
			acc_map = torch.sum(weights, -1)
			rgb_map = rgb_map + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

		return rgb_map, depth_map, weights.sum(dim=-1)

	def get_weights(self, density: torch.tensor, distance: torch.tensor):
		"""
		:param density: b x n x 1 array
		:param distance: b x n x 1 array
		:return: b x n x 1 array
		"""
		fe = distance * density  # free_energy
		alpha = 1. - torch.exp(-fe)
		shifted_fe = torch.cat([torch.zeros(*distance.shape[:-1], 1).to(fe.device), fe[..., :-1]], dim=-1)
		t = torch.exp(-torch.cumsum(shifted_fe, dim=-1))
		weights = alpha * t
		return weights

	def forward(self, inputs: dict) -> torch.tensor:
		"""
		:param pts: b x n x 3 locations of sampled points on the camera ray
		:param view_dirs: b x n x 3, viewing directions
		return: b x n x 1, density
		"""
		# Parse model input

		intrinsics = inputs["intrinsics"].to(self.device)
		uv = inputs["uv"].to(self.device)
		pose = inputs["pose"].to(self.device)
		depth = None
		# ray_dirs: b x n x 3
		# camera_loc: b x 3
		ray_dirs, cam_loc_ori = get_camera_params(uv, pose, intrinsics)
		batch_size, num_pixels, _ = ray_dirs.shape

		if "rasterizer" in inputs:
			depth = inputs["depth"].to(self.device)


		cam_loc = cam_loc_ori.unsqueeze(-2).repeat(1, num_pixels, 1)  # b x 1 x 3 -> b x n x 3 -> (bn ,3)
		ray_dirs = F.normalize(ray_dirs.reshape(batch_size, -1, 3), dim=-1) # bn x 3
		self.sdf.bias = self.density_func.get_beta().detach() #* (2 ** 0.5)
		if self.training:
			range= 0.05
		else:
			range= 0.1
		with torch.profiler.record_function("sampling"):
			z_vals = self.ray_sampler.get_z_vals(ray_dirs,
			                                     cam_loc,
			                                     SDF=self.sdf,
			                                     D=self.density_func,
			                                     surface= self.sdf.mesh if self.training else depth,
			                                     training=self.sdf.training,
			                                     r=range,
			                                     )


		N_samples = z_vals.shape[-1]

		pts = cam_loc.unsqueeze(-2) + z_vals.unsqueeze(-1) * ray_dirs.unsqueeze(-2)  # bn x 1 x 3 + bn x ns x 1 * bn x 1 x 3
		pts = pts.reshape(batch_size, -1, 3)  # in world coordinate b*n_cam x n_sample x 3

		dirs = ray_dirs.unsqueeze(-2).repeat(1, 1, N_samples, 1)
		dirs = dirs.reshape(batch_size, -1, 3)

		sd_vals, normals = self.sdf(pts)


		check_pcd = False
		if check_pcd:
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(pts.view(-1, 3).detach().cpu().numpy())
			colors = ((sd_vals > 0).to(torch.float64)*0.5 + 0.4).view(-1, 1).repeat(1, 3).detach().cpu().numpy()
			pcd.colors = o3d.utility.Vector3dVector(colors)
			if torch.sum(depth > 1e-3) > 0:
				o3d.io.write_point_cloud(f'sample_points_near_surf_{iter}.ply', pcd)
			# b, n_c * n_s , 1
		import matplotlib.pyplot as plt
		plot = 0
		if plot:
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')
			ax.scatter(self.sdf.mesh.get_vertices()[::50, 0].detach().cpu().numpy(),
			           self.sdf.mesh.get_vertices()[::50, 1].detach().cpu().numpy(),
			           self.sdf.mesh.get_vertices()[::50, 2].detach().cpu().numpy(),
			           s=1, c='m')
			ax.scatter(pts[0, ::10, 0].detach().cpu().numpy(),
			           pts[0, ::10, 1].detach().cpu().numpy(),
			           pts[0, ::10, 2].detach().cpu().numpy(),
			           c=torch.sign(sd_vals[0, ::10]).detach().cpu().numpy(),
			           s=1)
			plt.show()
			plt.close(fig)

		density = self.density_func(sd_vals).reshape(batch_size,
		                                             num_pixels,
		                                             N_samples,
		                                             1)  # b*n_c * n_s , 1
		# sample points around surface
		color = self.color_network(pts, dirs, normals).reshape(batch_size, num_pixels, N_samples, 3)  # b*n_c*n_s, 3
		# bn_c x n_s
		rgb_map, depth_map, mask = self.rendering_func(density, color, z_vals)

		if self.training:
			output = {
				'rgb_map': rgb_map,  # b x n x 3
				'beta': self.density_func.get_beta().detach().cpu().numpy(),
				'mesh': self.sdf.mesh,
				'mask': mask,
			}
		else:
			output = {
				'rgb_map': rgb_map * (inputs["depth"]>0).to(self.device).float(),# mask.unsqueeze(-1), #* (inputs["depth"]>0).to(self.device).float(),  # b x n x 3
				'depth_map': depth_map,
				'mask': (inputs["depth"]>0).to(self.device).float()
			}

		return output
