from scipy.stats import gennorm
from scipy.special import gamma
import torch
import math
#samples from a generalized  gaussian distribution, with variance 1
def randgn_like_cpu(tensor, p=2, device='cuda'):
	raw_variance = gamma(3./p)/gamma(1./p)
	raw_std = math.sqrt(raw_variance)
	flat =  gennorm.rvs(p, size=torch.numel(tensor))/raw_std
	return torch.tensor(flat).type(torch.float).cuda().reshape(tensor.shape)

# Same as above, but uses GPU. Derives generalized Gaussian distribution from Gamma Distribution,
# using technique described by:
# Nardon, Martina, and Paolo Pianca. "Simulation techniques for generalized Gaussian densities."
# Journal of Statistical Computation and Simulation 79.11 (2009): 1317-1329.
def randgn_like(tensor, p=2, device='cuda'):
	raw_variance = gamma(3./p)/gamma(1./p)
	raw_std = math.sqrt(raw_variance)
	b= raw_std**p
	a = 1./p
	m = torch.distributions.Gamma(torch.tensor(a).cuda(), torch.tensor(b).cuda())
	gmas = m.sample(tensor.shape)
	m2 = torch.distributions.Bernoulli(torch.tensor(0.5).cuda())
	signs = 1-2*m2.sample(tensor.shape)
	return torch.pow(gmas,a)*signs



