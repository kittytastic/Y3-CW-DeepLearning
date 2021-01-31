# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% id="MK1Jl7nkLnPA"
# this code is based on [ref], which is released under the MIT licesne
# make sure you reference any code you have studied as above here

# imports
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# hyperparameters
batch_size  = 256
latent_size = 32
dataset = 'cifar10'
#dataset = 'stl10'

# %%
import Utils
#import importlib
#importlib.reload(Utils)
from Utils import *

# %%
targetGPU = "GeForce GTX 1080 Ti"

if torch.cuda.is_available():
    targetDeviceNumber = None

    print("There are %d available GPUs:"%torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        prefix = "    "
        if torch.cuda.get_device_name(i) == targetGPU:
            targetDeviceNumber = i
            prefix = "[🔥]"

        print("%s %s"%(prefix, torch.cuda.get_device_name(i)))

    if targetDeviceNumber != None:
        device = torch.device('cuda:%d'%targetDeviceNumber)
    else:
        torch.device('cuda')
        raise Exception("Cannot find target GPU")
else:
    device = torch.device('cpu')
    raise Exception("Not using GPU")

# %% id="bK383zeDM4Ac"
image_size = None
image_channels = None
# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# you may use cifar10 or stl10 datasets
if dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('./Dataset/cifar10', train=True, download=True, transform=torchvision.transforms.Compose([
            #torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ])),
        shuffle=True, batch_size=batch_size, drop_last=True
    )

    image_channels = 3
    image_size=32
    class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# stl10 has larger images which are much slower to train on. You should develop your method with CIFAR-10 before experimenting with STL-10
if dataset == 'stl10':
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.STL10('./Dataset/stl10', split='train+unlabeled', download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])),
    shuffle=True, batch_size=batch_size, drop_last=True)
    
    image_channels = 3
    image_size=96
    class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'] # these are slightly different to CIFAR-10

train_iterator = iter(cycle(train_loader))

# %% colab={"base_uri": "https://localhost:8080/", "height": 630} id="BtJs-qxHRLXz" outputId="770d6f89-c5dc-4b2d-fca4-0d7b96ec256e"
# let's view some of the training data
x,t = next(train_iterator)
x,t = x.to(device), t.to(device)
plotTensor(x)


# %%
def CheckpointModel(model, checkpoint_name, epoch, loss):
    torch.save({'model':model.state_dict(), 'optimiser':model.optimiser.state_dict(), 'epoch':epoch, 'loss':loss}, '%s.chkpt'%checkpoint_name)

def RestoreModel(model, checkpoint_name):
    params = torch.load('%s.chkpt'%checkpoint_name)
    model.load_state_dict(params['model'])
    model.optimiser.load_state_dict(params['optimiser'])
    epoch = params['epoch']
    loss = params['loss']
    return epoch, loss


# %%
from tqdm import trange

def TrainModel(model, total_epoch, start_epoch=0, iter_count=len(train_loader)):

    epoch_iter = trange(start_epoch, total_epoch)
    epoch_loss = []
    t_mmd = []
    t_recon = []

    for epoch in epoch_iter:
        
        iter_loss = np.zeros(0)
        loss_item = None
        mmd_item = None
        recon_item = None
        

        for i in range(iter_count):
            # Get Data
            x,t = next(train_iterator)
            x,t = x.to(device), t.to(device)

            # Step Model
            loss, mmd_loss, recon_loss = model.forwardStep(x)
            model.backpropagate(loss)
            
            # Collect Stats
            loss_item = loss.detach().item()
            mmd_item = mmd_loss.detach().mean().cpu()
            recon_item = recon_loss.detach().mean().cpu()

            iter_loss = np.append(iter_loss, loss_item)

        
        
        epoch_loss.append(iter_loss.mean())
        t_mmd.append(mmd_item)
        t_recon.append(recon_item)

        # Print Status
        epoch_iter.set_description("Current Loss %.5f    Epoch" % loss_item)

    return (epoch_loss, t_mmd, t_recon)


# %%
def PlotRandomLatentSample(model, count=8):
    rand_latent = model.getRandomSample()
    plotTensor(rand_latent[0:8])

def PlotReconstructionAttempt(model):
    x,t = next(train_iterator)
    #x = x[0:8]
    x = x.to(device)
    x_hat = model.encode(x)
    x_hat = model.decode(x_hat)
    x = x[0:8]
    x_hat = x_hat[0:8]
    plotTensor(x_hat)
    plotTensor(x)

def CompareByExample(model1, model2):
    x,t = next(train_iterator)
    #x = x[0:8]
    x = x.to(device)
    x_hat_a = model1.encode(x)
    x_hat_a = model1.decode(x_hat_a)
    x_hat_b = model2.encode(x)
    x_hat_b = model2.decode(x_hat_b)
    plotTensor(x[0:8])
    plotTensor(x_hat_a[0:8])
    plotTensor(x_hat_b[0:8])


# %% [markdown] id="Qnjh12UbNFpV"
# **Define resnet VAE**

# %% tags=[]
import warnings

class VAE(nn.Module):
    def __init__(self, encoder, decoder, learning_rate = 1e-4):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

        # output size depends on input size for some encoders
        demo_input = torch.ones([batch_size, image_channels, image_size, image_size])
        h_dim = self.encoder(demo_input).shape[1]
        
        if h_dim < latent_size:
            warnings.warn('latent dimension [%d] > encoded dimension [%d]', latent_size, h_dim)

        # Resize encoding to latent size 
        self.encode_to_z = nn.Linear(h_dim, latent_size)

        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)
       


    def getRandomSample(self, count=batch_size):
        z = torch.rand(count, latent_size)
        z = z.to(device)
        return self.decode(z)

    def encode(self, x):
        x = self.encoder(x)
        x = self.encode_to_z(x)
        return x
  
    def decode(self, x):
        return self.decoder(x)

    def backpropagate(self, loss):
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    
    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd


    def forwardStep(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)

        random_z_sample = torch.randn(200, latent_size, device=device)

        mmd = self.compute_mmd(random_z_sample, z)
        nll = (x_hat - x).pow(2).mean()
        loss = nll + mmd

        return loss, mmd, nll


#print(f'> Number of VAE parameters {len(torch.nn.utils.parameters_to_vector(VAE().parameters()))}')


# %% [markdown]
# # Test Model

# %%
import ResNet
import importlib
importlib.reload(ResNet)
from ResNet import FCCResNet18Encoder, FCCResNet18Decoder


vae_enc = FCCResNet18Encoder(latent_size)
vae_dec = FCCResNet18Decoder(latent_size, image_size)
Vfcres = VAE(vae_enc, vae_dec).to(device)
all_loss, mmd_loss, recon_loss = TrainModel(Vfcres, 1320)
PlotAllLoss([all_loss, mmd_loss, recon_loss], ["Loss", "MMD", "Recon"])
PlotLoss(all_loss)

# %%
CheckpointModel(Vfcres, 'Vfcres-11hr', 1320, {'loss':all_loss, 'mmd':mmd_loss, 'recon':recon_loss})

# %%
PlotRandomLatentSample(Vfcres)

# %%
PlotReconstructionAttempt(Vfcres)

# %%
PlotLatentSpace(Vfcres, train_iterator, device, class_names)

# %%
import ResNet
import importlib
importlib.reload(ResNet)
from ResNet import ResNet18Encoder, ResNet18Decoder

vae_enc_old = ResNet18Encoder()
vae_dec_old = ResNet18Decoder(
    latent_dim=latent_size,
    input_height=image_size
)
Vres_old = VAE(vae_enc_old, vae_dec_old).to(device)
#elo_loss, kl_loss, recon_loss = TrainModel(Vres, 0)
#PlotAllLoss([elo_loss, kl_loss, recon_loss], ["EBLO", "KL", "Recon"])
#PlotLoss(elo_loss)
