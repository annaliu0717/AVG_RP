import torch as T
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import imageio, skimage, cv2

device = 'cuda' if T.cuda.is_available() else 'cpu'


def get_rays(H, W, focal, c2w):
	"""
	:param H: image height
	:param W: image width
	:param focal: focal length
	:param c2w: transformation matrix from camera to world
	:return: camera ray for each pixel
	"""
	w = T.from_numpy(np.arange(W)).float()  # Wx1
	h = T.from_numpy(np.arange(H)).float()  # Hx1
	x, y = T.meshgrid(w, h)  # wxh, wxh
	x = (x - W * .5) / focal[0]  # wxh
	y = -(y - H * .5) / focal[1]  # wxh
	z = -T.ones_like(x)  # wxh
	dirs = T.stack([x, y, z], dim=-1)  # wxhx3
	rays_d = T.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
	rays_o = c2w[:3, -1].expand(*rays_d.size())
	return rays_o, rays_d


def get_cv_raydir(pixelcoords, height, width, focal, rot):
	# pixelcoords: H x W x 2
	if isinstance(focal, float):
		focal = [focal, focal]
	x = (pixelcoords[..., 0] - width / 2.0) / focal[0]
	y = (pixelcoords[..., 1] - height / 2.0) / focal[1]
	z = np.ones_like(x)
	dirs = np.stack([x, y, z], axis=-1)
	dirs = np.sum(rot[None, None, :, :] * dirs[..., None], axis=-2)  # 1*1*3*3   x   h*w*3*1
	dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5)

	return dirs


def get_camera_rotation(eye, center, up):
	nz = center - eye
	nz /= np.linalg.norm(nz)
	x = np.cross(nz, up)
	x /= np.linalg.norm(x)
	y = np.cross(x, nz)
	return np.array([x, y, -nz]).T


def get_blender_raydir(pixelcoords, height, width, focal, rot, dir_norm):
	## pixelcoords: H x W x 2
	x = (pixelcoords[..., 0] + 0.5 - width / 2.0) / focal
	y = (pixelcoords[..., 1] + 0.5 - height / 2.0) / focal
	z = np.ones_like(x)
	dirs = np.stack([x, -y, -z], axis=-1)
	dirs = np.sum(dirs[..., None, :] * rot[:, :], axis=-1)  # h*w*1*3   x   3*3
	if dir_norm:
		# print("dirs",dirs-dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5))
		dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5)
	# print("dirs", dirs.shape)

	return dirs


def get_dtu_raydir(pixelcoords, intrinsic, rot, dir_norm):
	# rot is c2w
	## pixelcoords: H x W x 2
	x = (pixelcoords[..., 0] + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
	y = (pixelcoords[..., 1] + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
	z = np.ones_like(x)
	dirs = np.stack([x, y, z], axis=-1)
	# dirs = np.sum(dirs[...,None,:] * rot[:,:], axis=-1) # h*w*1*3   x   3*3
	dirs = dirs @ rot[:, :].T  #
	if dir_norm:
		# print("dirs",dirs-dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5))
		dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5)
	# print("dirs", dirs.shape)

	return dirs


def get_optix_raydir(pixelcoords, height, width, focal, eye, center, up):
	c2w = get_camera_rotation(eye, center, up)
	return get_blender_raydir(pixelcoords, height, width, focal, c2w)


def flip_z(poses):
	z_flip_matrix = np.eye(4, dtype=np.float32)
	z_flip_matrix[2, 2] = -1.0
	return np.matmul(poses, z_flip_matrix[None, ...])


def get_sphere_intersections(cam_loc, ray_dirs, r=1.0):
	# Input: b x n_rays x 3 ; b x n_rays x 3
	# Output: n_rays x 1, n_rays x 1 (close and far)

	ray_cam_dot = T.einsum('bnij, bnjk->bnik',
	                       ray_dirs.unsqueeze(-2),
	                       cam_loc.unsqueeze(-2).permute(0, 1, 3, 2)).squeeze(-1)  # b x n x 1
	under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(p=2, dim=-1, keepdim=True) ** 2 - r ** 2)

	# sanity check
	if (under_sqrt <= 0).sum() > 0:
		('BOUNDING SPHERE PROBLEM!')
		exit()

	sphere_intersections = T.sqrt(under_sqrt) * T.Tensor([-1, 1]).to(cam_loc.device).float() - ray_cam_dot
	sphere_intersections = sphere_intersections.clamp_min(0.0)

	return sphere_intersections


def get_camera_params(uv, pose, intrinsics):
	"""
	:param uv: pixel coordinates in camera
	:param pose: n x 7, rotation (4), x, y, z or n x 4 x 4
	:param intrinsics:
	:return:
	"""
	if pose.shape[1] == 7:  # In case of quaternion vector representation
		cam_loc = pose[:, 4:]
		R = quat_to_rot(pose[:, :4])
		p = T.eye(4).repeat(pose.shape[0], 1, 1).to(uv.device).float()
		p[:, :3, :3] = R
		p[:, :3, 3] = cam_loc
	else:  # In case of pose matrix representation
		cam_loc = pose[:, :3, 3]
		p = pose

	batch_size, num_samples, _ = uv.shape

	depth = T.ones((batch_size, num_samples)).to(uv.device)
	x_cam = uv[:, :, 0].view(batch_size, -1)
	y_cam = uv[:, :, 1].view(batch_size, -1)
	z_cam = depth.view(batch_size, -1)

	pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

	# permute for batch matrix product
	pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

	world_coords = T.bmm(p, pixel_points_cam).permute(0, 2, 1)[..., :3]  # pixel loc in world coordinate
	ray_dirs = world_coords - cam_loc[:, None, :]  # ray dir from camera_loc to pixel point in world coordinate
	ray_dirs = F.normalize(ray_dirs, dim=2)

	return ray_dirs, cam_loc


def get_camera_for_plot(pose):
	if pose.shape[1] == 7:  # In case of quaternion vector representation
		cam_loc = pose[:, 4:].detach()
		R = quat_to_rot(pose[:, :4].detach())
	else:  # In case of pose matrix representation
		cam_loc = pose[:, :3, 3]
		R = pose[:, :3, :3]
	cam_dir = R[:, :3, 2]
	return cam_loc, cam_dir

def point2uv(point, intrinsics, pose):
	"""
	:param point: n x 3
	:param intrinsic: b x 4 x 4
	:param pose: b x 4 x 4
	:return:
	"""
	b, d1, d2 = intrinsics.size()
	cam_loc = (pose[:, :3, 3]).unsqueeze(-1)
	R_T = (pose[:, :3, :3]).permute(0, 2, 1).unsqueeze(1)
	n, _ = point.size()
	# point = T.cat((point, T.ones(point.shape[0], 1).to(point.device)), -1) # n x 4
	point = point.expand(b, n, 3)[..., None]
	cam = T.matmul(R_T, point - cam_loc.unsqueeze(1))
	cam = T.cat((cam, T.ones(b, n, 1, 1).to(point.device)), -2)
	# k = T.bmm(intrinsics, pose)
	res = T.matmul(intrinsics[:, :3, :].unsqueeze(1), cam)
	res = res / (res[..., -1, :]).unsqueeze(-2)
	return (res[:, :, :2, :]).squeeze(-1)


def lift(x, y, z, intrinsics):
	# parse intrinsics
	fx = intrinsics[:, 0, 0]
	fy = intrinsics[:, 1, 1]
	cx = intrinsics[:, 0, 2]
	cy = intrinsics[:, 1, 2]
	sk = intrinsics[:, 0, 1]

	x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1) - sk.unsqueeze(
		-1) * y / fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
	y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

	# homogeneous
	return T.stack((x_lift, y_lift, z, T.ones_like(z).to(z.device)), dim=-1)


def quat_to_rot(q):
	batch_size, _ = q.shape
	q = F.normalize(q, dim=1)
	R = T.ones((batch_size, 3, 3)).cuda()
	qr = q[:, 0]
	qi = q[:, 1]
	qj = q[:, 2]
	qk = q[:, 3]
	R[:, 0, 0] = 1 - 2 * (qj ** 2 + qk ** 2)
	R[:, 0, 1] = 2 * (qj * qi - qk * qr)
	R[:, 0, 2] = 2 * (qi * qk + qr * qj)
	R[:, 1, 0] = 2 * (qj * qi + qk * qr)
	R[:, 1, 1] = 1 - 2 * (qi ** 2 + qk ** 2)
	R[:, 1, 2] = 2 * (qj * qk - qi * qr)
	R[:, 2, 0] = 2 * (qk * qi - qj * qr)
	R[:, 2, 1] = 2 * (qj * qk + qi * qr)
	R[:, 2, 2] = 1 - 2 * (qi ** 2 + qj ** 2)
	return R


def rot_to_quat(R):
	batch_size, _, _ = R.shape
	q = T.ones((batch_size, 4)).cuda()

	R00 = R[:, 0, 0]
	R01 = R[:, 0, 1]
	R02 = R[:, 0, 2]
	R10 = R[:, 1, 0]
	R11 = R[:, 1, 1]
	R12 = R[:, 1, 2]
	R20 = R[:, 2, 0]
	R21 = R[:, 2, 1]
	R22 = R[:, 2, 2]

	q[:, 0] = T.sqrt(1.0 + R00 + R11 + R22) / 2
	q[:, 1] = (R21 - R12) / (4 * q[:, 0])
	q[:, 2] = (R02 - R20) / (4 * q[:, 0])
	q[:, 3] = (R10 - R01) / (4 * q[:, 0])
	return q


def get_psnr(img1, img2, normalize_rgb=False):
	if normalize_rgb:  # [-1,1] --> [0,1]
		img1 = (img1 + 1.) / 2.
		img2 = (img2 + 1.) / 2.

	mse = T.mean((img1 - img2) ** 2)
	psnr = -10. * T.log(mse) / T.log(T.Tensor([10.]).to(img1.device))

	return psnr


def load_rgb(path, normalize_rgb=False):
	img = imageio.imread(path)
	img = skimage.img_as_float32(img)

	if normalize_rgb:  # [-1,1] --> [0,1]
		img -= 0.5
		img *= 2.
	img = img.transpose(2, 0, 1)
	return img


def load_K_Rt_from_P(filename, P=None):
	if P is None:
		lines = open(filename).read().splitlines()
		if len(lines) == 4:
			lines = lines[1:]
		lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
		P = np.asarray(lines).astype(np.float32).squeeze()

	out = cv2.decomposeProjectionMatrix(P)
	K = out[0]
	R = out[1]
	t = out[2]

	K = K / K[2, 2]
	intrinsics = np.eye(4)
	intrinsics[:3, :3] = K

	pose = np.eye(4, dtype=np.float32)
	pose[:3, :3] = R.transpose()
	pose[:3, 3] = (t[:3] / t[3])[:, 0]

	return intrinsics, pose
