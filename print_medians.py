import torch
import pandas 
import numpy as np
import statsmodels.stats.proportion
f = lambda x: statsmodels.stats.proportion.proportion_confint(x,100000,alpha=.002,method='beta')[0]
for noise in ["0.12","0.25","0.50","1.00"]:
	for p in [1,2,3,4,5,6]:
		csv = pandas.read_csv('data/certify/cifar10/resnet110/noise_'+noise+'_p_'+str(p)+'/test/sigma_'+noise+'_p_'+str(p),delimiter='\t')
		medgg = np.median(csv['generalized_gaussian_bound_over_sqrt(c)'].to_numpy()*csv['correct'].to_numpy())
		mediid = np.median(csv['any_iid_distribution_bound'].to_numpy()*csv['correct'].to_numpy())
		medcount = np.median(csv['count'].to_numpy()*csv['correct'].to_numpy())
		print('Median Count: 32*32, p:' + str(p) + ' noise:' + noise+ ' ===== ' + str(medcount))
		print('p_1 lower bound: 32*32, p:' + str(p) + ' noise:' + noise+ ' ===== ' + str(f(medcount)))
		print('IID bound size: 32*32, p:' + str(p) + ' noise:' + noise+ ' ===== ' + str(mediid))
		print('GG bound size: 32*32, p:' + str(p) + ' noise:' + noise+ ' ===== ' + str(medgg))
		if (p == 2):
			medgaus = np.median(csv['exact_radius'].to_numpy()*csv['correct'].to_numpy())
			print('Gaussian bound: 32*32, p:' + str(p) + ' noise:' + noise+ ' ===== ' + str(medgaus))
for scale in ["16","8"]:
	for noise in ["0.12","0.25","0.50","1.00"]:
		for p in [1,2,3,4,5,6]:
			csv = pandas.read_csv('data/certify/cifar10/resnet110/noise_'+noise+'_p_'+str(p)+'_scale_'+scale+'/test/sigma_'+noise+'_p_'+str(p)+'_scale_'+scale,delimiter='\t')
			medgg = np.median(csv['generalized_gaussian_bound_over_sqrt(c)'].to_numpy()*csv['correct'].to_numpy())
			mediid = np.median(csv['any_iid_distribution_bound'].to_numpy()*csv['correct'].to_numpy())
			medcount = np.median(csv['count'].to_numpy()*csv['correct'].to_numpy())
			print('Median Count: '+scale+'*'+scale+', p:' + str(p) + ' noise:' + noise+ ' ===== ' + str(medcount))
			print('p_1 lower bound: '+scale+'*'+scale+', p:' + str(p) + ' noise:' + noise+ ' ===== ' + str(f(medcount)))
			print('IID bound size: '+scale+'*'+scale+', p:' + str(p) + ' noise:' + noise+ ' ===== ' + str(mediid))
			print('GG bound size: '+scale+'*'+scale+', p:' + str(p) + ' noise:' + noise+ ' ===== ' + str(medgg))
			if (p == 2):
				medgaus = np.median(csv['exact_radius'].to_numpy()*csv['correct'].to_numpy())
				print('Gaussian bound: '+scale+'*'+scale+', p:' + str(p) + ' noise:' + noise+ ' ===== ' + str(medgaus))

