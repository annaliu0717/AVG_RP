import pytorch3d.structures
import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import trimesh
import pymeshlab

from pytorch3d import ops
from models.points import Points
import torch.nn.functional as F
from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix

from utils.mesh_utils import get_o3d_mesh

WORLD_UNIT=0.015729


class Mesh(nn.Module):
	def __init__(self,
	             vertices,
	             faces,
	             lam=10,
	             device='cpu'):
		super().__init__()
		self.device=device
		self.vertices = vertices.to(device)
		self.faces = nn.Parameter(faces.to(torch.long), requires_grad=False).to(device)
		self.l = lam
		M = compute_matrix(self.vertices, self.faces, lambda_=self.l)
		u = to_differential(M, self.vertices)

		self.u = nn.Parameter(u, requires_grad=True).to(device)
		self.M = nn.Parameter(M, requires_grad=False).to(device)
		self.update()


		#self.mesh_to_points()
		self.n_faces = self.faces.shape[0]
		self.n_vert =  self.vertices.shape[0]

	def get_faces(self):
		return self.faces

	def get_face_verts(self):
		return self.get_vertices()[self.faces.to(torch.long), :]

	def get_face_centroid(self):
		return self.face_centroid

	def get_vertices(self):
		return self.vertices

	def get_vert_normals(self):
		return self.vertex_normals

	def compute_face_centroid(self):
		face_vert = self.get_face_verts()
		return face_vert.mean(dim=-2).float()

	def compute_face_normal(self, triangles):
		a = triangles[..., 1, :] - triangles[..., 0, :]
		b = triangles[..., 2, :] - triangles[..., 0, :]
		return F.normalize(torch.cross(a, b, dim=-1), dim=-1)

	def compute_normals(self):
		# Compute the face normals
		triangles = self.get_face_verts()
		self.face_normals = self.compute_face_normal(triangles)
		#self.face_normals.requires_grad_(True)
		# Compute the vertex normals
		vertex_normals = torch.zeros_like(self.get_vertices())
		vertex_normals = vertex_normals.index_add(0, self.faces[..., 0], self.face_normals)
		vertex_normals = vertex_normals.index_add(0, self.faces[..., 1], self.face_normals)
		vertex_normals = vertex_normals.index_add(0, self.faces[..., 2], self.face_normals)
		self.vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1).float()
		self.vertex_normals.requires_grad_(True)

	def compute_edges(self, remove_duplicates=True):
		edges_0 = torch.index_select(self.get_faces(), 1, torch.tensor([0, 1], device=self.device))
		edges_1 = torch.index_select(self.get_faces(), 1, torch.tensor([1, 2], device=self.device))
		edges_2 = torch.index_select(self.get_faces(), 1, torch.tensor([2, 0], device=self.device))

		# Merge the into one tensor so that the three edges of one face appear sequentially
		# edges = [f0_e0, f0_e1, f0_e2, ..., fN_e0, fN_e1, fN_e2]
		edges = torch.cat([edges_0, edges_1, edges_2], dim=1).view(self.faces.shape[0] * 3, -1)

		if remove_duplicates:
			edges, _ = torch.sort(edges, dim=1)
			self.edges = torch.unique(edges, dim=0)

	def compute_avg_edge_length(self):
		verts_edges = self.get_vertices()[self.edges]
		v0, v1 = verts_edges.unbind(1)
		return (v0 - v1).norm(dim=1, p=2).mean()

	def mesh_to_points(self):
		self.vertices = Points(points=self.vertices, center=self.center,
		                       normal=self.vertex_normals,
		                       device=self.device)

	def gather_faces(self, idx):
		return self.faces[..., idx, :]

	def gather_vert_normals(self, idx):
		return self.vertex_normals[..., idx, :]

	def gather_vert_features(self, idx):
		return self.vert_features[..., idx, :]

	def update(self):
		if not self.M.is_coalesced():
			self.M = nn.Parameter(self.M.coalesce(), requires_grad=False)

		self.vertices = from_differential(self.M, self.u, 'Cholesky')
		self.face_centroid = self.compute_face_centroid()
		self.compute_normals()
		self.compute_edges()


	def load(self, u, M, faces):
		self.u = nn.Parameter(u).to(self.device)
		if not M.is_coalesced():
			M = M.coalesce()

		self.M = nn.Parameter(M, requires_grad=False)
		self.vertices = from_differential(self.M, self.u, 'Cholesky')
		self.faces = nn.Parameter(faces.to(torch.long), requires_grad=False)
		self.face_centroid = self.compute_face_centroid()
		self.compute_normals()
		self.n_faces = self.faces.shape[0]
		self.n_vert = self.vertices.shape[0]

	def get_cam_to_vert_rays(self, cam_loc):
		return self.get_vertices().unsqueeze(0) - cam_loc[:, 0, :].unsqueeze(-2)

	def get_near_bounds(self, range, creat_mesh=True):
		range = self.vertex_normals * range
		if creat_mesh:
			near_mesh = pytorch3d.structures.Meshes(verts=self.get_vertices().unsqueeze(0) + range,
			                                        verts_normals=self.vertex_normals.unsqueeze(0),
			                                        faces=self.get_faces().unsqueeze(0))

			return near_mesh
		else:
			return self.get_vertices() + range #, self.get_vertices() - range
	# def get_near_far_bounds(self, range, ray_dir, cam_loc):
	# 	range = self.vertex_normals * range
	# 	vertices_near = self.get_vertices() + range
	# 	vertices_far = self.get_vertices() - range
	#
	# 	bounds_near = ray_triangle_intersection(cam_loc, ray_dir,
	# 	                                        vertices_near[self.faces.to(torch.long), :],
	# 	                                        far=-1)
	# 	bounds_far = ray_triangle_intersection(cam_loc, ray_dir,
	# 	                                      vertices_far[self.faces.to(torch.long), :],
	# 	                                      far=-1)
	#
	# 	return bounds_near, bounds_far




	def remesh(self, target_len=0.012):
		ms = pymeshlab.MeshSet()
		vertices = self.get_vertices().to(torch.float64).detach().cpu().numpy()
		faces = self.get_faces().to(torch.float64).detach().cpu().numpy()
		v_normals_matrix = self.get_vert_normals().to(torch.float64).detach().cpu().numpy()
		mesh = pymeshlab.Mesh(vertex_matrix=vertices,
		                      face_matrix=faces,
		                      v_normals_matrix=v_normals_matrix)
		ms.add_mesh(mesh)
		target_len=pymeshlab.AbsoluteValue(target_len)
		ms.meshing_remove_duplicate_faces()
		ms.meshing_remove_duplicate_vertices()
		ms.meshing_repair_non_manifold_edges()
		ms.meshing_repair_non_manifold_vertices()
		ms.meshing_close_holes()
		ms.meshing_re_orient_faces_coherentely()
		ms.meshing_isotropic_explicit_remeshing(targetlen=target_len)
		ms.meshing_re_orient_faces_coherentely()
		# ms.apply_coord_hc_laplacian_smoothing()

		mesh = ms.current_mesh()
		vertices = torch.from_numpy(mesh.vertex_matrix()).float().to(self.device).detach()
		faces = torch.from_numpy(mesh.face_matrix()).float().to(self.device)
		ms.clear()
		M = compute_matrix(vertices, faces, lambda_=self.l)
		u = to_differential(M, vertices)
		self.faces = nn.Parameter(faces.to(torch.long), requires_grad=False).to(self.device)

		self.u = nn.Parameter(u, requires_grad=True).to(self.device)
		self.M = nn.Parameter(M, requires_grad=False).to(self.device)
		self.update()

		# self.mesh_to_points()
		self.n_faces = self.faces.shape[0]
		self.n_vert = self.vertices.shape[0]

	def sample_points(self, n_samples):
		mesh_o3d = get_o3d_mesh(self)
		mesh_o3d.sample_points_poisson_disk(number_of_points=n_samples)
		support_points = torch.from_numpy(np.array(mesh_o3d.vertices)).to(self.device).unsqueeze(0).float()
		normals = torch.from_numpy(np.array(mesh_o3d.vertex_normals)).to(self.device).float()
		self.support_points = support_points
		self.support_normals = normals

if __name__ == '__main__':
	mesh = Mesh(device='cuda:0')
	print(mesh.n_faces)
	print(mesh.n_vert)
	vert_i = torch.randint(mesh.n_vert, size=(3, 20))
	mesh.find_face(vert_i)