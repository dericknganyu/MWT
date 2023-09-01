"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

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



print("torch version is ",torch.__version__)
ntrain = 1000
ntest = 5000


learning_rate = 0.001

step_size = 100
gamma = 0.5

modes = 12
width = 32
ich = 3
initializer = get_initializer('xavier_normal')

parser = argparse.ArgumentParser(description='parse batch_size, epochs and resolution')
parser.add_argument('--bs',  default=10, type = int, help='batch-size')
parser.add_argument('--ep', default=500, type = int, help='epochs')
parser.add_argument('--res', default=512, type = int, help='resolution')
parser.add_argument('--wd', default=1e-4, type = float, help='weight decay')

args = parser.parse_args()

batch_size = args.bs #100
epochs = args.ep #500
res = args.res + 1#32#sys.argv[1]
wd = args.wd
print("\nUsing batchsize = %s, epochs = %s, and resolution = %s\n"%(batch_size, epochs, res))
params = {}
params["xmin"] = 0
params["ymin"] = 0
params["xmax"] = 1
params["ymax"] = 1


################################################################
# training and evaluation
################################################################
model = MWT2d(ich, 
            alpha = 12,
            c = 4,
            k = 4, 
            base = 'legendre', # 'chebyshev'
            nCZ = 4,
            L = 0,
            initializer = initializer,
            ).to(device)
print("Model has %s parameters"%(count_params(model)))
print("Model has %s parameters"%(count_params(model)))

PATH = "../../../../../../localdata/Derick/stuart_data/Darcy_421/new_aUP_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
resultPATH = ""
#Read Datasets
Y_train, X_train, Y_test, X_test = readtoArray(PATH, 1024, 5000, Nx = 512, Ny = 512)

print ("Converting dataset to numpy array.")
tt = time.time()
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print ("    Conversion completed after %.2f minutes"%((time.time()-tt)/60))

print ("Subsampling dataset to the required resolution.")
tt = time.time()
X_train = SubSample(X_train, res, res)[:, :-1, :-1]
Y_train = SubSample(Y_train, res, res)[:, :-1, :-1]
res = res-1
print ("    Subsampling completed after %.2f minutes"%((time.time()-tt)/60))

print ("Taking out the required train/test size.")
tt = time.time()
x_train = torch.from_numpy(X_train[:ntrain, :, :]).float()
y_train = torch.from_numpy(Y_train[:ntrain, :, :]).float()
print ("    Taking completed after %s seconds"%(time.time()-tt))
print("...")


x_normalizer = UnitGaussianNormalizer(x_train)

y_normalizer = UnitGaussianNormalizer(y_train)
y_normalizer.cuda()
#res= res+1
####TO EDIT
if res == 512:
	test_l2 = 0.01795
	timestamp = '20220810-080537'
	batch_size = 5
if res == 256:
    test_l2 = 0.043759
    timestamp = '20220806-111103'
if res == 128:
    test_l2 = 0.083692
    timestamp = '20220806-065603'
if res == 64:
    test_l2 = 0.109779
    timestamp = '0.109779'
if res == 32:
    test_l2 = 0.157347
    timestamp = '20220805-213833'
if res == 16:
    test_l2 = 0.20535
    timestamp = '20220805-210416'


#####
#res = res-1

ModelInfos = "_%03d"%(res)+"~res_"+str(np.round(test_l2,6))+"~RelL2TestError_"+str(ntrain)+"~ntrain_"+str(ntest)+"~ntest_"+str(batch_size)+"~BatchSize_"+str(learning_rate)+\
            "~LR_"+str(wd)+"~Reg_"+str(gamma)+"~gamma_"+str(step_size)+"~Step_"+str(epochs)+"~epochs_"+timestamp
          

dataLoss = np.loadtxt('files/inv/lossData'+ModelInfos+'.txt')

stepTrain = dataLoss[:,0] #Reading Epoch                  
errorTrain = dataLoss[:,1] #Reading erros
errorTest  = dataLoss[:,2]

print("Ploting Loss VS training step...")
fig = plt.figure(figsize=(15, 10))
plt.yscale('log')
plt.plot(stepTrain, errorTrain, label = 'Training Loss')
plt.plot(stepTrain , errorTest , label = 'Test Loss')
plt.xlabel('epochs')#, fontsize=16, labelpad=15)
plt.ylabel('Loss')
plt.legend(loc = 'upper right')
plt.title("lr = %s test error = %s"%(learning_rate, str(np.round(test_l2,6))))
plt.savefig("figures/inv/Error_VS_TrainingStep"+ModelInfos+".png", dpi=500)


#def use_model():#(params, model,device,nSample,params):

model.load_state_dict(torch.load("files/inv/last_model"+ModelInfos+".pt"))
model.eval()


print()
print()

#Just a file containing data sampled in same way as the training and test dataset
fileName = "new_aUP_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
U_train, F_train, U_test, F_test= readtoArray(fileName, 1, 1, 512, 512)
res = res+1
F_train = SubSample(np.array(F_train), res, res)[:, :-1, :-1]

print("Starting the Verification with Sampled Example")
tt = time.time()
U_FDM = SubSample(np.array(U_train), res, res)[0, :-1, :-1]
res = res-1
grids = []
grids.append(np.linspace(0, 1, res))
grids.append(np.linspace(0, 1, res))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,res,res,2)
grid = torch.tensor(grid, dtype=torch.float)

for i in range(25):
    print("      Doing MWT on Example...")
    tt = time.time()
    ff = torch.from_numpy(F_train).float()
    ff = x_normalizer.encode(ff)
    ff = torch.cat([ff.reshape(1,res,res,1), grid.repeat(1,1,1,1)], dim=3).cuda() #ff.reshape(1,res,res,1).cuda()#

    U_MWT = model(ff)
    U_MWT = U_MWT.reshape(1, res, res)
    U_MWT = y_normalizer.decode(U_MWT)
    U_MWT = U_MWT.detach().cpu().numpy()
    U_MWT = U_MWT[0] 
    print("            MWT completed after %.4f"%(time.time()-tt))

myLoss = LpLoss(size_average=False)
print()
print("Ploting comparism of FDM and MWT Simulation results")
fig = plt.figure(figsize=((5+2)*4, 5))

fig.suptitle(r"Plot of $-\nabla \cdot (a(s) \nabla u(s)) = f(s), \partial \Omega = 0$ with $u|_{\partial \Omega}  = 0.$")

colourMap = parula() #plt.cm.jet #plt.cm.coolwarm

plt.subplot(1, 4, 1)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Input")
plt.imshow(F_train[0], cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 2)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("FDM")
plt.imshow(U_FDM, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 3)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("MWT")
plt.imshow(U_MWT, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 4)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("FDM-MWT, RelL2Err = "+str(round(myLoss.rel_single(U_MWT, U_FDM).item(), 3)))
plt.imshow(np.abs(U_FDM - U_MWT), cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.savefig('figures/inv/1compare'+ModelInfos+'.png',dpi=500)

#plt.show()
fig = plt.figure(figsize=((5+1)*2, 5))

plt.subplot(1, 2, 1)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("MWT")
plt.imshow(U_MWT, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 2, 2)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("FDM-MWT, RelL2Err = "+str(round(myLoss.rel_single(U_MWT, U_FDM).item(), 3)))
plt.imshow(np.abs(U_FDM - U_MWT), cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

fig.tight_layout()
plt.savefig('figures/inv/MWT-Darcy-Inverse-UP.png',dpi=500)