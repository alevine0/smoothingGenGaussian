# Curse of Dimensionality on Randomized Smoothing for Certifiable Robustness

This repository contains code for the paper "Curse of Dimensionality on Randomized Smoothing for Certifiable Robustness" by Anonymous Authors. The code is forked from the publically-availabe code from the paper [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/abs/1902.02918) by (Cohen et al. 2019).

Example training for 16-by-16 cifar-10 with generalized Gaussian noise for shape parameter q=3 and noise standard deviation 0.5:
```
python  code/train.py cifar10 cifar_resnet110  models/cifar10/resnet110/noise_0.50_p_3_scale_16 --batch 400 --noise 0.50 --p 3 --scale_down 2

```

Example certification:
```
mkdir data/certify/cifar10/resnet110/noise_0.50_p_${P}_scale_16
mkdir data/certify/cifar10/resnet110/noise_0.50_p_${P}_scale_16/test
python code/certify.py cifar10 models/cifar10/resnet110/noise_0.50_p_3_scale_16/checkpoint.pth.tar 0.50 data/certify/cifar10/resnet110/noise_0.50_p_3_scale_16/test/sigma_0.50_p_3_scale_16 --skip 20 --p 3 --scale_down 2 --batch 400

```
Additionally, once all certificates are generated ``print_medians.py`` compiles statistics used in our figures.