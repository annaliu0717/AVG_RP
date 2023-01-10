import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.ops as ops
from models.mesh import Mesh
from utils.mesh_utils import sample_points_from_meshes, get_p3d_mesh
import numpy as np
import open3d as o3d

class SDF(nn.Module):
	def __init__(self, mesh: Mesh, s=0.2, k=1):  # sigma=0.1):
		super(SDF, self).__init__()
		self.mesh = mesh
		self.k = k
		self.device=mesh.device
		self.s = nn.Parameter(torch.tensor(s).float().to(self.device),
		                      requires_grad=False)
		self.bias = 0.002
		# self.n_samples = 49999

	def forward(self, pts: torch.tensor, k=0):
		"""
		:param pts: b*n_cam* n_samples, 3
		:param k:
		:return:
		"""
		pts.requires_grad_(True)
		if len(pts.size()) < 3:
			pts = pts.unsqueeze(0)

		#if support_pts is None:
		#	support_pts = self.mesh.get_vertices().unsqueeze(0)

		if k == 0:
			k = self.k

		nn = ops.knn_points(pts,
		                    self.mesh.get_vertices().unsqueeze(0),
		                    K = k,
		                    return_nn=True,
		                    return_sorted=True)
		nn_normals_near = F.normalize(self.mesh.get_vert_normals()[..., nn.idx, :], dim=-1)

		if self.training:
			nn.knn.requires_grad_(True)
			nn_normals_near.requires_grad_(True)

			convex = F.normalize((nn.knn[..., 1:, :] - nn.knn[..., 0:1, :]), dim=-1)
			convex = (convex * nn_normals_near[..., 0:1, :]).sum(dim=-1, keepdim=True)
			convex = convex.mean(dim=-2, keepdim=True)
			convex[convex.abs() < 1e-8] = 0.

			knn = nn.knn + nn_normals_near * convex.sign() * self.bias
		else:
			knn = nn.knn

		p = pts.unsqueeze(-2) - knn # b x n_p x k x 3
		inside = ((p * nn_normals_near).sum(dim=-1) < 0).float()[..., None] # b x n_p x 3 x 1
		nn_normals_far = -p * inside + (1 - inside) * p
		del inside
		dists = p.norm(dim=-1)
		w_d = 1 / (dists + 1e-5)
		w_n = 0.1

		nn_normals_far = F.normalize(nn_normals_far, dim=-1)

		w_p = torch.minimum(dists, torch.exp(-self.s * w_d))
		h = (p * (
						((w_n * nn_normals_near) + (w_p.unsqueeze(-1) * nn_normals_far)) / (w_n + w_p + 1e-5).unsqueeze(-1))
		     ).sum(dim=-1)

		sdf = (w_d * h).sum(dim=-1, keepdim=True) / w_d.sum(dim=-1, keepdim=True)
		normals = (w_d.unsqueeze(-1) * nn_normals_near).sum(dim=-2) / (w_d).sum(dim=-1, keepdim=True)


		return sdf.squeeze(-1), F.normalize(normals, dim=-1)


	def get_nn(self):
		return self.nn_points

	def get_nn_idx(self):
		return self.nn_idx
