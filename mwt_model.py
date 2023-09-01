import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utils import *


import time
from datetime import datetime
from readData import readtoArray
import os, sys
from colorMap import parula
import argparse

from utils_mwt import *
import math as math
from torch import Tensor
from typing import List, Tuple

torch.manual_seed(0)
np.random.seed(0)

################################################################
# Wavelet layer
################################################################
 
 
class sparseKernel2d(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernel2d,self).__init__()
        
        self.k = k
        self.conv = self.convBlock(k, c*k**2, alpha)
        self.Lo = nn.Linear(alpha*k**2, c*k**2)
        
    def forward(self, x):
        B, Nx, Ny, c, ich = x.shape # (B, Nx, Ny, c, k**2)
        x = x.view(B, Nx, Ny, -1)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, c, ich)
        
        return x
        
        
    def convBlock(self, k, W, alpha):
        och = alpha * k**2
        net = nn.Sequential(
            nn.Conv2d(W, och, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        return net 

class sparseKernelFT2d(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernelFT2d, self).__init__()        
        
        self.modes = alpha

        self.weights1 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, dtype=torch.cfloat))        
        nn.init.xavier_normal_(self.weights1)
        nn.init.xavier_normal_(self.weights2)
        
        self.Lo = nn.Linear(c*k**2, c*k**2)
        self.k = k
    
    def compl_mul2d(self, x, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", x, weights)
        
    def forward(self, x):
        B, Nx, Ny, c, ich = x.shape # (B, N, N, c, k^2)
        
        x = x.view(B, Nx, Ny, -1)
        x = x.permute(0, 3, 1, 2)
        x_fft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        l1 = min(self.modes, Nx//2+1)
        l1l = min(self.modes, Nx//2-1)
        l2 = min(self.modes, Ny//2+1)
        out_ft = torch.zeros(B, c*ich, Nx, Ny//2 + 1,  device=x.device, dtype=torch.cfloat)
        
        out_ft[:, :, :l1, :l2] = self.compl_mul2d(
            x_fft[:, :, :l1, :l2], self.weights1[:, :, :l1, :l2])
        out_ft[:, :, -l1:, :l2] = self.compl_mul2d(
                x_fft[:, :, -l1:, :l2], self.weights2[:, :, :l1, :l2])
        
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s = (Nx, Ny))
        
        x = x.permute(0, 2, 3, 1)
        x = F.relu(x)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, c, ich)
        return x

class MWT_CZ2d(nn.Module):
    def __init__(self,
                 k = 3, alpha = 5, 
                 L = 0, c = 1,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT_CZ2d, self).__init__()
        
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0@PHI0
        G0r = G0@PHI0
        H1r = H1@PHI1
        G1r = G1@PHI1
        H0r[np.abs(H0r)<1e-8]=0
        H1r[np.abs(H1r)<1e-8]=0
        G0r[np.abs(G0r)<1e-8]=0
        G1r[np.abs(G1r)<1e-8]=0

        self.A = sparseKernelFT2d(k, alpha, c)
        self.B = sparseKernel2d(k, c, c)
        self.C = sparseKernel2d(k, c, c)
        
        self.T0 = nn.Linear(c*k**2, c*k**2)

        if initializer is not None:
            self.reset_parameters(initializer)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((np.kron(H0, H0).T, 
                            np.kron(H0, H1).T,
                            np.kron(H1, H0).T,
                            np.kron(H1, H1).T,
                           ), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((np.kron(G0, G0).T,
                            np.kron(G0, G1).T,
                            np.kron(G1, G0).T,
                            np.kron(G1, G1).T,
                           ), axis=0)))
        
        self.register_buffer('rc_ee', torch.Tensor(
            np.concatenate((np.kron(H0r, H0r), 
                            np.kron(G0r, G0r),
                           ), axis=0)))
        self.register_buffer('rc_eo', torch.Tensor(
            np.concatenate((np.kron(H0r, H1r), 
                            np.kron(G0r, G1r),
                           ), axis=0)))
        self.register_buffer('rc_oe', torch.Tensor(
            np.concatenate((np.kron(H1r, H0r), 
                            np.kron(G1r, G0r),
                           ), axis=0)))
        self.register_buffer('rc_oo', torch.Tensor(
            np.concatenate((np.kron(H1r, H1r), 
                            np.kron(G1r, G1r),
                           ), axis=0)))
        
        
    def forward(self, x):
        
        B, Nx, Ny, c, ich = x.shape # (B, Nx, Ny, c, k**2)
        ns = math.floor(np.log2(Nx))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

#         decompose
        for i in range(ns-self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x.view(B, 2**self.L, 2**self.L, -1)).view(
            B, 2**self.L, 2**self.L, c, ich) # coarsest scale transform

#        reconstruct            
        for i in range(ns-1-self.L,-1,-1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)

        return x

    
    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2 , ::2 , :, :], 
                        x[:, ::2 , 1::2, :, :], 
                        x[:, 1::2, ::2 , :, :], 
                        x[:, 1::2, 1::2, :, :]
                       ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s
        
        
    def evenOdd(self, x):
        
        B, Nx, Ny, c, ich = x.shape # (B, Nx, Ny, c, k**2)
        assert ich == 2*self.k**2
        x_ee = torch.matmul(x, self.rc_ee)
        x_eo = torch.matmul(x, self.rc_eo)
        x_oe = torch.matmul(x, self.rc_oe)
        x_oo = torch.matmul(x, self.rc_oo)
        
        x = torch.zeros(B, Nx*2, Ny*2, c, self.k**2, 
            device = x.device)
        x[:, ::2 , ::2 , :, :] = x_ee
        x[:, ::2 , 1::2, :, :] = x_eo
        x[:, 1::2, ::2 , :, :] = x_oe
        x[:, 1::2, 1::2, :, :] = x_oo
        return x
    
    def reset_parameters(self, initializer):
        initializer(self.T0.weight)
    

class MWT2d(nn.Module):
    def __init__(self,
                 ich = 1, k = 3, alpha = 2, c = 1,
                 nCZ = 3,
                 L = 0,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT2d,self).__init__()
        
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk = nn.Linear(ich, c*k**2)
        
        self.MWT_CZ = nn.ModuleList(
            [MWT_CZ2d(k, alpha, L, c, base, 
            initializer) for _ in range(nCZ)]
        )
        self.Lc0 = nn.Linear(c*k**2, 128)
        self.Lc1 = nn.Linear(128, 1)
        
        if initializer is not None:
            self.reset_parameters(initializer)
        
    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        
        B, Nx, Ny, ich = x.shape # (B, Nx, Ny, d)
        ns = math.floor(np.log2(Nx))
        x = self.Lk(x)
        x = x.view(B, Nx, Ny, self.c, self.k**2)
    
        for i in range(self.nCZ):
            x = self.MWT_CZ[i](x)
            if i < self.nCZ-1:
                x = F.relu(x)

        x = x.view(B, Nx, Ny, -1) # collapse c and k**2
        x = self.Lc0(x)
        x = F.relu(x)
        x = self.Lc1(x)
        return x.squeeze()
    
    def reset_parameters(self, initializer):
        initializer(self.Lc0.weight)
        initializer(self.Lc1.weight)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)



def get_initializer(name):
    
    if name == 'xavier_normal':
        init_ = partial(nn.init.xavier_normal_)
    elif name == 'kaiming_uniform':
        init_ = partial(nn.init.kaiming_uniform_)
    elif name == 'kaiming_normal':
        init_ = partial(nn.init.kaiming_normal_)
    return init_



class sparseKernel1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernel1d,self).__init__()
       
        self.k = k
        self.Li = nn.Linear(c*k, 128)
        self.conv = self.convBlock(c*k, 128)
        self.Lo = nn.Linear(128, c*k)
       
    def forward(self, x):
        B, N, c, ich = x.shape # (B, N, c, k)
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.Lo(x)
        x = x.view(B, N, c, ich)
        return x
       
       
    def convBlock(self, ich, och):
        net = nn.Sequential(
            nn.Conv1d(ich, och, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        return net

def compl_mul1d(x, weights):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", x, weights)

class sparseKernelFT1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernelFT1d, self).__init__()       
       
        self.modes1 = alpha
        self.scale = (1 / (c*k*c*k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c*k, c*k, self.modes1, dtype=torch.cfloat))
        self.weights1.requires_grad = True
        self.k = k
       
    def forward(self, x):
        B, N, c, k = x.shape # (B, N, c, k)
       
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N//2+1)
        out_ft = torch.zeros(B, c*k, N//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x


class MWT_CZ1d(nn.Module):
    def __init__(self,
                 k = 3, alpha = 5,
                 L = 0, c = 1,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT_CZ1d, self).__init__()
       
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0@PHI0
        G0r = G0@PHI0
        H1r = H1@PHI1
        G1r = G1@PHI1
        
        H0r[np.abs(H0r)<1e-8]=0
        H1r[np.abs(H1r)<1e-8]=0
        G0r[np.abs(G0r)<1e-8]=0
        G1r[np.abs(G1r)<1e-8]=0
       
        self.A = sparseKernelFT1d(k, alpha, c)
        self.B = sparseKernelFT1d(k, alpha, c)
        self.C = sparseKernelFT1d(k, alpha, c)
       
        self.T0 = nn.Linear(k, k)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))
       
        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))
       
       
    def forward(self, x):
       
        B, N, c, ich = x.shape # (B, N, k)
        ns = math.floor(np.log2(N))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
#         decompose
        for i in range(ns-self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x) # coarsest scale transform

#        reconstruct           
        for i in range(ns-1-self.L,-1,-1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        return x

   
    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                       ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s
       
       
    def evenOdd(self, x):
       
        B, N, c, ich = x.shape # (B, N, c, k)
        assert ich == 2*self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)
       
        x = torch.zeros(B, N*2, c, self.k,
            device = x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x
   
   
class MWT1d(nn.Module):
    def __init__(self,
                 ich = 1, k = 3, alpha = 2, c = 1,
                 nCZ = 3,
                 L = 0,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT1d,self).__init__()
       
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk = nn.Linear(ich, c*k)
       
        self.MWT_CZ = nn.ModuleList(
            [MWT_CZ1d(k, alpha, L, c, base,
            initializer) for _ in range(nCZ)]
        )
        self.Lc0 = nn.Linear(c*k, 128)
        self.Lc1 = nn.Linear(128, 1)
       
        if initializer is not None:
            self.reset_parameters(initializer)
       
    def forward(self, x):
       
        B, N, ich = x.shape # (B, N, d)
        ns = math.floor(np.log2(N))
        x = self.Lk(x)
        x = x.view(B, N, self.c, self.k)
   
        for i in range(self.nCZ):
            x = self.MWT_CZ[i](x)
            if i < self.nCZ-1:
                x = F.relu(x)

        x = x.view(B, N, -1) # collapse c and k
        x = self.Lc0(x)
        x = F.relu(x)
        x = self.Lc1(x)
        return x.squeeze()
   
    def reset_parameters(self, initializer):
        initializer(self.Lc0.weight)
        initializer(self.Lc1.weight)
        
        
