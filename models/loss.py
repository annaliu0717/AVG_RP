import pytorch3d.structures
from pytorch3d.loss import mesh_normal_consistency
import torch
from torch import nn
import torch.nn.functional as F

class RenderLoss(nn.Module):
	def __init__(self, mask_loss, weight_rgb=1.0, weight_normal=0.1, weight_mask=0.5):
		super(RenderLoss, self).__init__()
		self.weight_rgb = weight_rgb
		self.weight_normal = weight_normal
		self.weight_mask = weight_mask
		self.rgb_loss_function = nn.L1Loss(reduction='mean')
		self.bce_loss = nn.BCELoss(reduction='mean')

		self.mask_loss = mask_loss()

	def forward(self, model_outputs: dict, ground_truth: torch.tensor):
		device = model_outputs['rgb_map'].get_device()
		rgb_gt = ground_truth['rgb'].to(device)
		mask_gt = ground_truth['mask'].to(device).float()
		mesh = model_outputs['mesh']
		mask = model_outputs['mask']
		mask = mask.clamp(1e-3, 1.0 - 1e-3)
		mask_loss_func2 = nn.MSELoss(reduction='mean')
		mask_loss = self.mask_loss(mask, mask_gt)
		if 'mask_rast' in model_outputs:
			mask_rast = model_outputs['mask_rast']
			mask_loss2 = mask_loss_func2(mask_rast, mask_gt)
		else:
			mask_loss2 = 0.

		idx = (mask + mask_gt) > 0

		mesh_p3d = pytorch3d.structures.Meshes(verts=mesh.get_vertices().unsqueeze(0),
		                                       faces=mesh.get_faces().unsqueeze(0),
		                                       verts_normals=mesh.get_vert_normals().unsqueeze(0))
		normal_consistency = mesh_normal_consistency(mesh_p3d)

		rgb_loss = self.rgb_loss(model_outputs['rgb_map'], rgb_gt, idx)


		return self.weight_rgb * rgb_loss +\
		       self.weight_normal * normal_consistency +\
		       self.weight_mask * (mask_loss + mask_loss2)

	def rgb_loss(self, rgb_pre, rgb_gt, idx=None):
		if idx is None:
			return self.rgb_loss_function(rgb_pre, rgb_gt)
		else:
			return self.rgb_loss_function(rgb_pre[..., idx, :], rgb_gt[..., idx, :])

	def normal_consistency_loss(self, mesh):
		""" Compute the normal consistency term as the cosine similarity between neighboring face normals.
		Args:
				mesh (Mesh): Mesh with face normals.
		"""

		loss = 1 - torch.cosine_similarity(mesh.face_normals[mesh.connected_faces[:, 0]],
		                                   mesh.face_normals[mesh.connected_faces[:, 1]], dim=1)
		return (loss ** 2).mean()


class FocalLoss(nn.Module):
	def __init__(self, gamma=2., alpha=1.):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha

	def forward(self, inputs, targets):
		BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
		pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
		F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
		return F_loss.mean()
