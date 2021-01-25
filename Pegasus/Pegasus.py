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
n_channels  = 3
latent_size = 512
dataset = 'cifar10'

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
# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# you may use cifar10 or stl10 datasets
if dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('./Dataset/cifar10', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])),
        shuffle=True, batch_size=batch_size, drop_last=True
    )
    #print(len(train_loader))
    class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# stl10 has larger images which are much slower to train on. You should develop your method with CIFAR-10 before experimenting with STL-10
if dataset == 'stl10':
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.STL10('./Dataset/stl10', split='train+unlabeled', download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])),
    shuffle=True, batch_size=batch_size, drop_last=True)
    train_iterator = iter(cycle(train_loader))
    class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'] # these are slightly different to CIFAR-10

train_iterator = iter(cycle(train_loader))

# %% colab={"base_uri": "https://localhost:8080/", "height": 630} id="BtJs-qxHRLXz" outputId="770d6f89-c5dc-4b2d-fca4-0d7b96ec256e"
# let's view some of the training data
plt.rcParams['figure.dpi'] = 175
x,t = next(train_iterator)
x,t = x.to(device), t.to(device)
plt.grid(False)
plt.title("Example data")
plt.imshow(torchvision.utils.make_grid(x).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
plt.show()

# %%
from tqdm import trange

def TrainModel(model, total_epoch, start_epoch=0):

    epoch_iter = trange(start_epoch, total_epoch)
    epoch_loss = []

    for epoch in epoch_iter:
        
        iter_loss = np.zeros(0)

        for i in range(len(train_loader)):
            # Get Data
            x,t = next(train_iterator)
            x,t = x.to(device), t.to(device)

            # Step Model
            loss = model.trainingStep(x, t)
            model.backpropagate(loss)
            
            # Collect Stats
            iter_loss = np.append(iter_loss, loss.item())

        
        #avg_loss = iter_loss.mean()
        epoch_loss.append(iter_loss[-1])

        # Print Status
        epoch_iter.set_description("Current Loss %.5f    Epoch" % loss.item())

    return epoch_loss


# %%
def PlotLoss(loss_array):
    plt.plot(loss_array)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()



# %%
def PlotModelRandomSample(model):
    rand_latent = model.getRandomSample()
    plt.rcParams['figure.dpi'] = 175
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(rand_latent).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.show()


# %%
# Plot Latent Space
import umap
import pandas as pd
import umap.plot

def PlotLatentSpace(model, point_count=1000):
    plot_count = 1000

    acc_labels = None
    acc_vals = None

    for i in range(plot_count//batch_size):
        # sample x from the dataset
        x,l = next(train_iterator)
        x,t = x.to(device), l.to(device)

        z = model.full_encode(x).cpu()

        latent_space = z.detach().numpy()
        labels = np.array(l)
        
        latent_space = np.squeeze(latent_space)

        if not acc_labels is None:
            acc_labels = np.concatenate((acc_labels, labels), axis=0)
            acc_vals = np.concatenate((acc_vals, latent_space))
        else:
            acc_labels = labels
            acc_vals = latent_space


    df = pd.DataFrame(data=acc_vals)

    mapper = umap.UMAP(n_neighbors=15).fit(df)
    df['labels'] = acc_labels
    df['labels'] = df['labels'].map(dict((i, val) for i, val in enumerate(class_names)))
    umap.plot.points(mapper, labels=df["labels"])


# %% [markdown] id="Qnjh12UbNFpV"
# **Define a simple convolutional autoencoder**

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="RGbLY6X-NH4O" outputId="e2f24af2-f398-42fc-b173-9f710147718e"
# simple block of convolution, batchnorm, and leakyrelu
class Block(nn.Module):
    def __init__(self, in_f, out_f):
        super(Block, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_f),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.f(x)

# define the model
class Autoencoder(nn.Module):
    def __init__(self, f=16):
        super().__init__()

        self.encode = nn.Sequential(
            Block(n_channels, f),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 16x16 (if cifar10, 48x48 if stl10)
            Block(f  ,f*2),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 8x8
            Block(f*2,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 4x4
            Block(f*4,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 2x2
            Block(f*4,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 1x1
            Block(f*4,latent_size),
        )

        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2), # output = 2x2
            Block(latent_size,f*4),
            nn.Upsample(scale_factor=2), # output = 4x4
            Block(f*4,f*4),
            nn.Upsample(scale_factor=2), # output = 8x8
            Block(f*4,f*2),
            nn.Upsample(scale_factor=2), # output = 16x16
            Block(f*2,f  ),
            nn.Upsample(scale_factor=2), # output = 32x32
            nn.Conv2d(f,n_channels, 3,1,1),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=1e-4)
    
    def getRandomSample(self):
        k = torch.rand(batch_size, latent_size, 1, 1)
        k = k.to(device)
        return self.decode(k)

    def full_encode(self, x):
        return self.encode(x)


    def backpropagate(self, loss):
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def trainingStep(self, x, t):
        # MSE
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = ((x-x_hat)**2).mean()
        return loss


print(f'> Number of autoencoder parameters {len(torch.nn.utils.parameters_to_vector(Autoencoder().parameters()))}')


# %% tags=[]
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class CondenseEncoder(nn.Module):
    def __init__(self, f=16):
        super().__init__()

        self.encode = nn.Sequential(
            Block(n_channels, f),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 16x16 (if cifar10, 48x48 if stl10)
            Block(f  ,f*2),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 8x8
            Block(f*2,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 4x4
            Block(f*4,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 2x2
            Block(f*4,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 1x1
            Block(f*4,latent_size),
            nn.Flatten()
        )
    
    def forward(self, x):
        return self.encode(x)

class CondenseDecoder(nn.Module):
    def __init__(self, f=16):
        super().__init__()

        self.decode = nn.Sequential(
            View((batch_size, latent_size, 1, 1)),
            nn.Upsample(scale_factor=2), # output = 2x2
            Block(latent_size,f*4),
            nn.Upsample(scale_factor=2), # output = 4x4
            Block(f*4,f*4),
            nn.Upsample(scale_factor=2), # output = 8x8
            Block(f*4,f*2),
            nn.Upsample(scale_factor=2), # output = 16x16
            Block(f*2,f  ),
            nn.Upsample(scale_factor=2), # output = 32x32
            nn.Conv2d(f,n_channels, 3,1,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decode(x)


class VAE(nn.Module):
    def __init__(self, f=16):
        super().__init__()

        self.encode = CondenseEncoder()
        self.decode = CondenseDecoder()

        # distribution parameters
        self.fc_mu = nn.Linear(latent_size, latent_size)
        self.fc_var = nn.Linear(latent_size, latent_size)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.optimiser = torch.optim.Adam(self.parameters(), lr=1e-4)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        
        # sum over last dim to go from single dim distribution to multi-dim
        kl = kl.sum(-1)
        return kl

    def getRandomSample(self):
        k = torch.rand(batch_size, latent_size)
        k = k.to(device)
        return self.decode(k)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def backpropagate(self, loss):
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def full_encode(self, x):
        x_encoded = self.encode(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z

    def trainingStep(self, x, t):

        # encode x to get the mu and variance parameters
        x_encoded = self.encode(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decode(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo_loss = (kl - recon_loss)
        elbo_loss = elbo_loss.mean()

        return elbo_loss


print(f'> Number of VAE parameters {len(torch.nn.utils.parameters_to_vector(VAE().parameters()))}')


# %% [markdown]
# ** VAE **

# %%
V = VAE().to(device)
elo_loss = TrainModel(V, 1)
PlotLoss(elo_loss)

# %%
PlotModelRandomSample(V)

# %%
PlotLatentSpace(V)

# %% [markdown] id="N1UBl0PJjY-f"
# **Auto Encoder**

# %% tags=[]
A = Autoencoder().to(device)
elo_loss = TrainModel(A, 5)
PlotLoss(elo_loss)

# %%
PlotModelRandomSample(A)

# %%
PlotLatentSpace(A)

# %% id="HjHPwDAe-YoI"
# optional example code to save your training progress for resuming later if you authenticated Google Drive previously
#torch.save({'A':A.state_dict(), 'optimiser':optimiser.state_dict(), 'epoch':epoch}, './checkpoint.chkpt')

# %% id="nrCN7YQ5-2J8"
# optional example to resume training if you authenticated Google Drive previously
#params = torch.load('./checkpoint.chkpt')
#A.load_state_dict(params['A'])
#optimiser.load_state_dict(params['optimiser'])
#epoch = params['epoch']
