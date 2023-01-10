import torch
import torch.nn as nn


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
	def __init__(self, beta=0.1, beta_min=0.0001):
		super().__init__()
		self.beta = nn.Parameter(torch.tensor(beta))
		self.beta_min = torch.tensor(beta_min)

	def forward(self, sd_val: torch.tensor, beta=None):
		if beta is None:
			beta = self.get_beta()
		alpha = 1 /  (beta).to(sd_val.device)
		density = alpha * (0.5 + 0.5 * torch.sign(sd_val) * torch.expm1(-sd_val.abs() / beta))
		if torch.isnan(density).sum() > 0:
			print(beta.min())
		return density

	def get_beta(self):
		beta = self.beta.abs() + self.beta_min
		return beta