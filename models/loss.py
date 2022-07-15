import torch
import torch as T
from torch import nn
import utils.general as utils

class RenderLoss(nn.Module):
	def __init__(self, alpha=2e-3):
		super(RenderLoss, self).__init__()
		self.alpha = alpha
	def forward(self, model_outputs: dict, ground_truth: T.tensor):
		device = model_outputs['rgb_map'].get_device()
		# uvs = model_outputs['uvs_near_surface']  # b x n x k x 2
		# uvs = uvs.view(uvs.shape[0], -1, 2)  # b x nk x 2
		rgb_gt = ground_truth['rgb'].to(device)
		# rgb_whole = ground_truth['rgb_whole'].to(device)
		confs = model_outputs['confs'] + 1e-4
		# rgb_syn = self.interpolation(uvs, rgb_whole)
		loss = nn.L1Loss(reduction='mean')
		l_sparse = (T.log(confs) + T.log(1 - confs)).sum() / confs.size().numel()
		rgb_loss = loss(model_outputs['rgb_map'], rgb_gt) + self.alpha * l_sparse
		# + loss(model_outputs['rgb_near_surface'], rgb_syn)
		return rgb_loss
	def interpolation(self, uvs, rgb_gt):
		pixel_coords_1 = T.stack([uvs[..., 0].floor(), uvs[..., 1].floor()], dim=-1).long()# x1 y1 b x n  x 2
		pixel_coords_2 = T.stack([uvs[..., 0].floor(), uvs[..., 1].ceil()], dim=-1).long()  # x1 y2
		pixel_coords_3 = T.stack([uvs[..., 0].ceil(), uvs[..., 1].floor()], dim=-1).long()  # x2 y1
		pixel_coords_4 = T.stack([uvs[..., 0].ceil(), uvs[..., 1].ceil()], dim=-1).long()  # x2 y2
		def gather_rgb(pix_coord, rgb):
			batch, w, h, d = rgb.size()
			_, n, _ = pix_coord.size()
			rgb_gt = torch.zeros(batch, n, d).to(rgb.device)
			for b in range(batch):
				pix_coord_valid_b = pix_coord[b, ((pix_coord[b, :, 0] < w) &
				                                  (pix_coord[b, :, 1] < h) &
				                                  (pix_coord[b, :, 0] >= 0) &
				                                  (pix_coord[b, :, 1] >= 0))]
				rgb_gt[b, (pix_coord[b, :, 0] < w) &
				          (pix_coord[b, :, 1] < h) &
				          (pix_coord[b, :, 0] >= 0) &
				          (pix_coord[b, :, 1] >= 0), :] = rgb[b,
				                                          pix_coord_valid_b[:, 0],
				                                          pix_coord_valid_b[:, 1], :]
				del pix_coord_valid_b
			return rgb_gt
		f_xy1 = (pixel_coords_3[..., 0] - uvs[..., 0]).unsqueeze(-1) * gather_rgb(pixel_coords_1, rgb_gt) +\
		         (uvs[..., 0] - pixel_coords_1[..., 0]).unsqueeze(-1) * gather_rgb(pixel_coords_3, rgb_gt)
		f_xy2 = (pixel_coords_3[..., 0] - uvs[..., 0]).unsqueeze(-1) * gather_rgb(pixel_coords_2, rgb_gt) +\
		         (uvs[..., 0] - pixel_coords_1[..., 0]).unsqueeze(-1) * gather_rgb(pixel_coords_4, rgb_gt)
		rgb_syn = (pixel_coords_2[..., 1] - uvs[..., 1]).unsqueeze(-1) * f_xy1 +\
		          (uvs[..., 1] - pixel_coords_1[..., 1]).unsqueeze(-1) * f_xy2
		return rgb_syn