import torch as T
import torch.nn as nn


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
	def __init__(self, beta=0.1, beta_min=0.0001):
		super().__init__()
		self.beta = nn.Parameter(T.tensor(beta))
		self.beta_min = T.tensor(beta_min)

	def forward(self, sd_val: T.tensor, beta=None):
		if beta is None:
			beta = self.get_beta()
		alpha = 1 / beta.to(sd_val.device)
		return alpha * (0.5 + 0.5 * T.sign(sd_val) * T.expm1(-sd_val.abs() * alpha))

	def get_beta(self):
		beta = self.beta.abs() + self.beta_min
		return beta