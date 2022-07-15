import torch as T
import torch.nn as nn
from pytorch3d import ops
from utils.render_utils import point2uv


class Points(nn.Module):
	def __init__(self,
	             points=None,
	             center=None,
	             n_points=10000,
	             feature_dim=9,
	             radius= 1.,
	             device='cpu'):
		super().__init__()
		if center is None:
			center = T.zeros(1, 3).float()
		if points is None:
			points = self.sphere_generator(n_points=n_points, radius=radius).to(device) + center.to(device)
			# points = T.cat((points, T.ones(n_points, 1).to(device) * 0.5), dim=-1) # confidence
			points = T.cat((points, T.zeros(n_points, feature_dim).to(device)), dim=-1)

		self.model = nn.Sequential(
          nn.Linear(3+feature_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 128),
          nn.ReLU(),
					nn.Linear(128, 1),
        )
		self.output_layer = nn.Sigmoid()
		# self.center = nn.Parameter(center).to(device)
		self.points = nn.Parameter(points).to(device)
		#self.normals = ops.estimate_pointcloud_normals((self.points+self.center).unsqueeze(0)).squeeze(0) # n x 3
		self.update_normals() # ops.estimate_pointcloud_normals(self.points.unsqueeze(0)).squeeze(0)
		self.uvs = None
		self.sample_pts = None

	def get_points_loc(self):
		return self.points[..., :3] # + self.center

	def get_confidence(self, points):
		# return self.points[..., 3]
		return self.output_layer(self.model(points))

	def get_features(self):
		# return self.points[..., 4:]
		return self.points[..., 3:]

	def get_points(self):
		return self.points

	def get_normals(self):
		self.update_normals()
		return self.normals

	def get_neighbours(self, loc, k=8):
		if len(loc.size()) != len(self.points.size()):
			b, n, d = loc.size()
			points = self.get_points_loc().expand(b, *self.get_points_loc().size())
		else:
			points = self.get_points_loc()
		assert points.requires_grad
		nn = ops.knn_points(loc, points, K=k, return_nn=True)
		nn_normals = self.get_normals()[..., nn.idx, :]
		nn_points = self.gather_points(nn.idx)
		return nn, nn_normals, nn_points

	def gather(self, idx):
		return self.get_points_loc()[..., idx, :]

	def gather_points(self, idx):
		return self.points[..., idx, :]

	def update_normals(self):
		if len(self.get_points_loc().size()) < 3:
			self.normals = ops.estimate_pointcloud_normals(self.get_points_loc().unsqueeze(0)).squeeze(0)
		else:
			self.normals = ops.estimate_pointcloud_normals(self.get_points_loc())

	def sphere_generator(self, n_points: int = 10000, radius: float = 1.) -> T.tensor:
		import numpy as np
		theta = 2 * np.pi * np.random.rand(n_points)
		phi = np.arccos(2 * np.random.rand(n_points) - 1)
		x = radius * np.cos(theta) * np.sin(phi)
		y = radius * np.sin(theta) * np.sin(phi)
		z = radius * np.cos(phi)
		return T.from_numpy(np.array([x,y,z]).T).float()

	def update_uv(self, intrinsics, pose):
		self.uvs = point2uv(self.get_points_loc(), intrinsics, pose).squeeze(-1) # b x n x 3

	def get_uvs(self, idx):
		return self.uvs[:, idx, :]

	def normal_sampler_around_pts(self, N_samples=64, std=1.0):
		self.sample_pts = std * T.randn(self.get_points_loc().shape[0], N_samples) + self.get_points_loc()[..., 0].unsqueeze(-1)


class PointsPruner():
	def __init__(self, t_gamma):
		self.t_gamma = t_gamma

	def prune(self, points):
		points.points = points.get_points()[points.get_confidence() >= self.t_gamma, ...]

class PointsGrower():
	def __init__(self, t_opa, t_dist):
		self.t_opa = t_opa
		self.t_dist = t_dist

	def grow(self,
	         points,
	         loc,
	         density,
	         features):
		new_point = T.cat((loc, density, features), dim=-1)
		points.points = T.cat((points.get_points(), new_point), dim=0)


