# Copyright (c) 2023 42dot. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F


def pack_cam_feat(x):
    """
    This function packs camera dimension to batch dimension.
    """
    if isinstance(x, dict):
        for k, v in x.items():
            b, n_cam = v.shape[:2]
            x[k] = v.view(b*n_cam, *v.shape[2:])
        return x
    else:
        b, n_cam = x.shape[:2]
        x = x.view(b*n_cam, *x.shape[2:])
    return x


def unpack_cam_feat(x, b, n_cam):
    """
    This function unpacks batch dimension into batch and camera dimensions.
    """    
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = v.view(b, n_cam, *v.shape[1:])
        return x
    else:
        x = x.view(b, n_cam, *x.shape[1:])
    return x


def upsample(x):
    """
    This function upsamples input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode='nearest')


def conv2d(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, nonlin = 'LRU', padding_mode = 'reflect', norm = False):
    """
    This function computes 2d convolutions followed by the specific nonlinear function(LeakyReLU, ELU, or None)
    """

    if nonlin== 'LRU':
        act = nn.LeakyReLU(0.1, inplace=True)
    elif nonlin == 'ELU':
        act = nn.ELU(inplace=True)
    else:
        act = nn.Identity()
        
    if norm:
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=False, padding_mode=padding_mode)        
        bnorm = nn.BatchNorm2d(out_planes)
    else:
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)        
        bnorm = nn.Identity()
    return nn.Sequential(conv, bnorm, act)


def conv1d(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, nonlin='LRU', padding_mode='reflect', norm = False):
    """
    This function computes 1d convolutions follwed by the nonlinear function(LeakyReLU, or None)
    """
    if nonlin== 'LRU':
        act = nn.LeakyReLU(0.1, inplace=True)
    elif nonlin == 'ELU':
        act = nn.ELU(inplace=True)    
    else:
        act = nn.Identity()
        
    if norm:
        conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=False, padding_mode=padding_mode)        
        bnorm = nn.BatchNorm1d(out_planes)
    else:
        conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)        
        bnorm = nn.Identity()        
        
    return nn.Sequential(conv, bnorm, act)        