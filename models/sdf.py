from models.points import Points
import torch as T
import torch.nn as nn


def dist_point_to_mesh(loc, p):
	b, n, k, d = p.size()
	assert k == 3
	n = T.cross(p[:, :, 2, :]-p[:, :, 0, :],
	            p[:, :, 1, :] - p[:, :, 0, :],
	            dim=-1)
	n = n / T.square((n ** 2).sum(dim=-1, keepdims=True))
	return ((loc - p[:, :, 0, :]) * n).sum(dim=-1, keepdims=True)


class SDF(nn.Module):
	def __init__(self, conf, points: Points):
		super(SDF, self).__init__()
		self.points = points
		self.k = conf.get_int("k")

	def forward(self, pts: T.tensor):
		"""
		:param pts: b*n_cam* n_samples, 3
		:param k:
		:return:
		"""
		nn, nn_normals, nn_points = self.points.get_neighbours(pts, self.k)  # b*n_c*n_s, k, :
		self.nn_points_loc = nn.knn # b*n_c*n_s, k, 3
		self.nn_normals = nn_normals
		self.nn_idx = nn.idx
		self.nn_dist = nn.dists
		self.nn_points = nn_points
		# self.nn_points.requires_grad_(True)
		# return dist_point_to_mesh(pts, self.nn_points)
		# self.nn_points.requires_grad_(True)
		w = 1 / self.nn_dist
		conf = self.points.get_confidence(nn_points)
		self.nn_conf = conf.squeeze(-1)
		w = w * self.nn_conf
		w = w / w.sum(dim=-1, keepdims=True)
		assert nn.dists.requires_grad
		signs = T.sign(((self.nn_points_loc - pts.unsqueeze(2)) * nn_normals).sum(dim=-1)) # b*n_c*n_s, k, k
		dists = (self.nn_dist * signs * w).sum(dim=-1)
		# dists[nn.dists.abs().min(-1).values < 1e-4] = 0
		return dists

	def get_nn(self):
		return self.nn_points

	def get_nn_idx(self):
		return self.nn_idx
