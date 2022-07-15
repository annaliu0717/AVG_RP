import os
from glob import glob
import torch as T


def mkdir_ifnotexists(directory):
	if not os.path.exists(directory):
		os.mkdir(directory)


def get_class(kls):
	parts = kls.split('.')
	module = ".".join(parts[:-1])
	m = __import__(module)
	for comp in parts[1:]:
		m = getattr(m, comp)
	return m


def glob_imgs(path):
	imgs = []
	for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
		imgs.extend(glob(os.path.join(path, ext)))
	return imgs


def split_input(model_input, total_pixels, n_pixels=10000, device='cpu'):
	'''
	 Split the input to fit Cuda memory for large resolution.
	 Can decrease the value of n_pixels in case of cuda out of memory error.
	 '''
	split = []
	for i, indx in enumerate(T.split(T.arange(total_pixels).to(device), n_pixels, dim=0)):
		data = model_input.copy()
		data['uv'] = T.index_select(model_input['uv'], 1, indx).to(device)
		if 'object_mask' in data:
			data['object_mask'] = T.index_select(model_input['object_mask'], 1, indx).to(device)
		split.append(data)
	return split


def merge_output(res, total_pixels, batch_size):
	''' Merge the split output. '''

	model_outputs = {}
	model_outputs['rgb_map'] = T.cat([r['rgb_map'].reshape(batch_size, -1, 3) for r in res],
	                                 1).reshape(batch_size * total_pixels, 3)
	model_outputs['sd_vals'] = T.cat([r['sd_vals'] for r in res],
	                                 0).reshape(-1, 1)
	model_outputs['samples'] = T.cat([r['samples'] for r in res],
	                                 0).reshape(-1, 3)
	return model_outputs


def concat_home_dir(path):
	return os.path.join(os.environ['HOME'], 'data', path)
