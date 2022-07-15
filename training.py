import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch as T
from tqdm import tqdm

import utils.general as utils
import utils.plot as plt
from utils import render_utils

from models.loss import RenderLoss


class TrainRunner():
	def __init__(self, **kwargs):
		T.set_default_dtype(T.float32)
		T.set_num_threads(1)

		self.conf = ConfigFactory.parse_file(kwargs['conf'])
		self.batch_size = kwargs['batch_size']
		self.nepochs = kwargs['nepochs']
		self.gpu_idx = kwargs['gpu_idx']
		self.exps_folder_name = kwargs['exps_folder_name']
		self.device = f'cuda:{self.gpu_idx}' if T.cuda.is_available() else 'cpu'
		self.expname = self.conf.get_string('train.expname') + kwargs['expname']
		scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
		if scan_id != -1:
			self.expname = self.expname + '_{0}'.format(scan_id)

		if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
			if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], self.expname)):
				timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], self.expname))
				if (len(timestamps)) == 0:
					is_continue = False
					timestamp = None
				else:
					timestamp = sorted(timestamps)[-1]
					is_continue = True
			else:
				is_continue = False
				timestamp = None
		else:
			timestamp = kwargs['timestamp']
			is_continue = kwargs['is_continue']

		utils.mkdir_ifnotexists(os.path.join('./', self.exps_folder_name))
		self.expdir = os.path.join('./', self.exps_folder_name, self.expname)
		utils.mkdir_ifnotexists(self.expdir)
		self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
		utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

		self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
		utils.mkdir_ifnotexists(self.plots_dir)

		# create checkpoints dirs
		self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
		utils.mkdir_ifnotexists(self.checkpoints_path)
		self.model_params_subdir = "ModelParameters"
		self.optimizer_params_subdir = "OptimizerParameters"
		self.scheduler_params_subdir = "SchedulerParameters"

		utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
		utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
		utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

		os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

		print('shell command : {0}'.format(' '.join(sys.argv)))

		print('Loading data ...')

		dataset_conf = self.conf.get_config('dataset')
		if kwargs['scan_id'] != -1:
			dataset_conf['scan_id'] = kwargs['scan_id']

		self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

		self.ds_len = len(self.train_dataset)
		print('Finish loading data. Data-set size: {0}'.format(self.ds_len))
		if scan_id < 24 and scan_id > 0:  # BlendedMVS, running for 200k iterations
			self.nepochs = int(200000 / self.ds_len)
			print('RUNNING FOR {0}'.format(self.nepochs))

		self.train_dataloader = T.utils.data.DataLoader(self.train_dataset,
		                                                batch_size=self.batch_size,
		                                                shuffle=True,
		                                                collate_fn=self.train_dataset.collate_fn)
		self.plot_dataloader = T.utils.data.DataLoader(self.train_dataset,
		                                               batch_size=self.conf.get_int('plot.plot_nimgs'),
		                                               shuffle=True,
		                                               collate_fn=self.train_dataset.collate_fn
		                                               )

		conf_model = self.conf.get_config('model')

		self.model = utils.get_class(self.conf.get_string('train.model_class'))(device=self.device,
		                                                                        conf=conf_model)
		                                                                        # center=self.train_dataset.center)
		self.model.to(self.device)

		self.loss = RenderLoss()

		self.lr = self.conf.get_float('train.learning_rate')
		self.optimizer = T.optim.Adam(self.model.parameters(), lr=self.lr)
		# Exponential learning rate scheduler
		decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
		decay_steps = self.nepochs * len(self.train_dataset)
		self.scheduler = T.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1. / decay_steps))

		self.do_vis = kwargs['do_vis']

		self.start_epoch = 0
		if is_continue:
			old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

			saved_model_state = T.load(
				os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
			self.model.load_state_dict(saved_model_state["model_state_dict"])
			self.start_epoch = saved_model_state['epoch']

			data = T.load(
				os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
			self.optimizer.load_state_dict(data["optimizer_state_dict"])

			data = T.load(
				os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
			self.scheduler.load_state_dict(data["scheduler_state_dict"])

		self.num_pixels = self.conf.get_int('train.num_pixels')
		self.total_pixels = self.train_dataset.total_pixels
		self.img_res = self.train_dataset.img_res
		self.n_batches = len(self.train_dataloader)
		self.plot_freq = self.conf.get_int('train.plot_freq')
		self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
		self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
		self.plot_conf = self.conf.get_config('plot')

	def save_checkpoints(self, epoch):
		T.save(
			{"epoch": epoch, "model_state_dict": self.model.state_dict()},
			os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
		T.save(
			{"epoch": epoch, "model_state_dict": self.model.state_dict()},
			os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

		T.save(
			{"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
			os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
		T.save(
			{"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
			os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

		T.save(
			{"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
			os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
		T.save(
			{"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
			os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

	def run(self):
		print("training...")

		for epoch in range(self.start_epoch, self.nepochs + 1):
			# if epoch % 2 == 0:
			plt.plot_points_iter(self.model.points.get_points_loc()[::10, ...].detach().cpu().numpy())

			if epoch % self.checkpoint_freq == 0:
				self.save_checkpoints(epoch)

			if self.do_vis and (epoch + 1) % self.plot_freq == 0:
				self.model.eval()


				self.train_dataset.change_sampling_idx(-1)
				indices, model_input, ground_truth = next(iter(self.plot_dataloader))

				model_input["intrinsics"] = model_input["intrinsics"].to(self.device)
				model_input["uv"] = model_input["uv"].to(self.device)
				model_input['pose'] = model_input['pose'].to(self.device)

				split = utils.split_input(model_input,
				                          self.total_pixels,
				                          n_pixels=self.split_n_pixels,
				                          device = self.device)
				res = []
				for s in tqdm(split):
					out = self.model(s)
					d = {'rgb_map': out['rgb_map'].detach(),
					     'sd_vals': out['sd_vals'].detach(),
					     'samples': out['samples'].detach()}
					res.append(d)

				batch_size = ground_truth['rgb'].shape[0]
				model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
				model_outputs['point_cloud'] = self.model.points.get_points_loc()[::50, ...].detach()
				plot_data = self.get_plot_data(model_outputs, model_input['pose'], ground_truth['rgb'])
				plt.plot(indices,
				         plot_data,
				         self.plots_dir,
			           epoch,
				         self.img_res,
			           self.plot_conf.get_int('plot_nimgs'))
				del plot_data, model_outputs, res
				self.model.train()

			self.train_dataset.change_sampling_idx(self.num_pixels)

			for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
				if T.cuda.is_available():
					T.cuda.empty_cache()
				model_input["intrinsics"] = model_input["intrinsics"].to(self.device)
				model_input["uv"] = model_input["uv"].to(self.device)
				model_input['pose'] = model_input['pose'].to(self.device)

				model_outputs = self.model(model_input)
				# plt.plot_points_iter(self.model.points.get_points()[::10, ...].detach().cpu().numpy())
				loss = self.loss(model_outputs, ground_truth)

				self.optimizer.zero_grad()
				if T.cuda.is_available():
					T.cuda.empty_cache()
				loss.backward(retain_graph=True)
				self.optimizer.step()

				psnr = render_utils.get_psnr(model_outputs['rgb_map'].reshape(-1, 3),
				                             ground_truth['rgb'].to(self.device).reshape(-1, 3))
				print(
					'{0}_{1} [{2}] ({3}/{4}): loss = {5}, psnr = {6}'
						.format(self.expname, self.timestamp, epoch, data_index, self.n_batches, loss.item(),
					          psnr.item()))

				self.train_dataset.change_sampling_idx(self.num_pixels)
				self.scheduler.step()

		self.save_checkpoints(epoch)

	def get_plot_data(self, model_outputs, pose, rgb_gt):
		batch_size, num_samples, _ = rgb_gt.shape

		rgb_eval = model_outputs['rgb_map'].reshape(batch_size, num_samples, -1)
		#normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
		#normal_map = (normal_map + 1.) / 2.

		plot_data = {
			'rgb_gt': rgb_gt,
			#'pose': pose,
			'rgb_eval': rgb_eval,
			'sd_vals': model_outputs['sd_vals'],
			'samples': model_outputs['samples'],
			'point_cloud': model_outputs['point_cloud']
		}

		return plot_data
