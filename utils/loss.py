#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


import numpy as np
import cv2
def image2canny(image, thres1, thres2, isEdge1=True):
    """ image: (H, W, 3)"""
    canny_mask = torch.from_numpy(cv2.Canny((image.detach().cpu().numpy()*255.).astype(np.uint8), thres1, thres2)/255.)
    if not isEdge1:
        canny_mask = 1. - canny_mask
    return canny_mask.float()

with torch.no_grad():
    kernelsize=3
    conv = torch.nn.Conv2d(1, 1, kernel_size=kernelsize, padding=(kernelsize//2))
    kernel = torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]]).reshape(1,1,kernelsize,kernelsize)
    conv.weight.data = kernel #torch.ones((1,1,kernelsize,kernelsize))
    conv.bias.data = torch.tensor([0.])
    conv.requires_grad_(False)
    conv = conv.cuda()


def nearMean_map(array, mask, kernelsize=3):
    """ array: (H,W) / mask: (H,W) """
    cnt_map = torch.ones_like(array)

    nearMean_map = conv((array * mask)[None,None])
    cnt_map = conv((cnt_map * mask)[None,None])
    nearMean_map = (nearMean_map / (cnt_map+1e-8)).squeeze()
        
    return nearMean_map