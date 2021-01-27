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
if dataset == 'fashion-mnist':
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('./Dataset/fashion-mnist', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x.convert('RGB') ),
            torchvision.transforms.ToTensor(),
        ])),
        shuffle=True, batch_size=batch_size, drop_last=True
    )

    image_channels = 3
    image_size = 28
    class_names = ['top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# you may use cifar10 or stl10 datasets
if dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('./Dataset/cifar10', train=True, download=True, transform=torchvision.transforms.Compose([
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
plt.rcParams['figure.dpi'] = 175
x,t = next(train_iterator)
x,t = x.to(device), t.to(device)
plt.grid(False)
plt.title("Example data")
plt.imshow(torchvision.utils.make_grid(x).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
plt.show()


# %%
def CheckpointModel(model, checkpoint_name, epoch):
    torch.save({'model':model.state_dict(), 'optimiser':model.optimiser.state_dict(), 'epoch':epoch}, '%s.chkpt'%checkpoint_name)

def RestoreModel(model, checkpoint_name):
    params = torch.load('%s.chkpt'%checkpoint_name)
    model.load_state_dict(params['model'])
    model.optimiser.load_state_dict(params['optimiser'])
    epoch = params['epoch']
    return epoch


# %%
from tqdm import trange

def TrainModel(model, total_epoch, start_epoch=0, iter_count=len(train_loader)):

    epoch_iter = trange(start_epoch, total_epoch)
    epoch_loss = []
    t_kl = []
    t_recon = []

    for epoch in epoch_iter:
        
        iter_loss = np.zeros(0)
        loss_item = None
        kl_item = None
        recon_item = None
        

        for i in range(iter_count):
            # Get Data
            x,t = next(train_iterator)
            x,t = x.to(device), t.to(device)

            # Step Model
            loss, kl_loss, recon_loss, _ = model.trainingStep(x, t)
            model.backpropagate(loss)
            
            # Collect Stats
            loss_item = loss.detach().item()
            kl_item = kl_loss.detach().mean().cpu()
            recon_item = recon_loss.detach().mean().cpu()

            iter_loss = np.append(iter_loss, loss_item)

        
        
        epoch_loss.append(iter_loss[-1])
        t_kl.append(kl_item)
        t_recon.append(recon_item)

        # Print Status
        epoch_iter.set_description("Current Loss %.5f    Epoch" % loss_item)

    return (epoch_loss, t_kl, t_recon)


# %%
def PlotLoss(loss_array, loss_type="Loss"):
    plt.plot(loss_array)
    plt.ylabel(loss_type)
    plt.xlabel('Epoch')
    plt.show()


# %%
def PlotAllLoss(losses, loss_names):
    fig, axs = plt.subplots(len(losses), sharex=True, gridspec_kw={'hspace': 0})
    for i in range(len(losses)):
        axs[i].plot(losses[i])
        axs[i].set_ylabel(loss_names[i])
    plt.xlabel('Epoch')
    plt.show()


# %%
def PlotModelRandomGeneratedSample(model):
    rand_latent = model.getRandomSample()
    plt.rcParams['figure.dpi'] = 175
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(rand_latent).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.show()


# %%
def PlotModelSampleEncoding(model):
    x,t = next(train_iterator)
    x = x.to(device)
    _, _, _, x_hat = model.trainingStep(x, t)
    plt.rcParams['figure.dpi'] = 175
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(x_hat[0:10]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.show()
    plt.imshow(torchvision.utils.make_grid(x[0:10]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.show()


# %%
def PlotSmallRandomSample(model, count=8):
    rand_latent = model.getRandomSample()
    plt.rcParams['figure.dpi'] = 175
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(rand_latent[0:10]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.show()


# %%
def HorseBirdTensors(count=batch_size):
    hc = 0
    bc = 0

    horses = torch.zeros(count, image_channels, image_size, image_size, requires_grad=False)
    birds = torch.zeros(count, image_channels, image_size, image_size, requires_grad=False)

    while hc<count or bc<count:
        x,t = next(train_iterator)
        for i in range(len(t)):
            cn = class_names[t[i].item()]
            if cn == "horse" and hc<count:
                horses[hc] = x[i]
                hc+=1
            if cn == "bird" and bc<count:
                birds[bc] = x[i]
                bc+=1
    return horses, birds

def plotTensor(tensor):
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(tensor).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.show()

def SeeHB(model):
    horses, birds = HorseBirdTensors(count=128)

    plotTensor(horses)
    plotTensor(birds)

def TryPegasus(model, width=8, rows=8):
    horses, birds = HorseBirdTensors(count=rows)

    gpu_horses = horses.to(device)
    gpu_birds = birds.to(device)
    z_horses = model.full_encode(gpu_horses)
    z_birds = model.full_encode(gpu_birds)
    
    
    z_amalgum = torch.zeros(rows*width, latent_size)
    for i in range(rows):
        cm = width-1
        for j in range(width):
            z_amalgum[i*width+j] = z_horses[i]*j/cm + z_birds[i]*(1-j/cm)

    z_amalgum = z_amalgum.to(device)
    pegasus_decoded = model.decode(z_amalgum)
    plotTensor(pegasus_decoded)

    plotTensor(horses)
    plotTensor(birds)
   

#TryPegasus(Vres2)

# %%
# Plot Latent Space
import umap
import pandas as pd
import umap.plot

def PlotLatentSpace(model, point_count=1000):

    acc_labels = None
    acc_vals = None

    for i in range(point_count//batch_size):
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

def PlotCustomLatentSpace(model, datasets, class_labels):
    full_df =  pd.DataFrame()
    data_df = pd.DataFrame()
    labels_df = pd.DataFrame()

    data = np.empty([0, latent_size])
    labels = []
    for i in range(len(datasets)):
        ds = datasets[i]
        l = class_labels[i]
        ds, l = ds.to(device), l

        z = model.full_encode(ds).detach().cpu().numpy()
        z = np.squeeze(z)
        data = np.concatenate((data, z), axis=0)
        labels+=[l]*len(ds)
    
    
    mapper = umap.UMAP(n_neighbors=15).fit(data)
    labels_df['labels']=labels
    umap.plot.points(mapper, labels=labels_df['labels'])


# %%
def InvestigateBirds(model, count=128):
    horses, birds = HorseBirdTensors(count=count)

    PlotCustomLatentSpace(model, [horses, birds], ['horses', 'birds'])

InvestigateBirds(Vres2, count=1024)

# %% [markdown] id="Qnjh12UbNFpV"
# **Define resnet VAE**

# %% tags=[]
from ResNet import resnet18_encoder, resnet18_decoder

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encode = encoder
        self.decode = decoder

        # output size depends on input size for some encoders
        demo_input = torch.ones([batch_size, image_channels, image_size, image_size])
        h_dim = self.encode(demo_input).shape[1]
        
        # distribution parameters
        self.fc_mu = nn.Linear(h_dim, latent_size)
        self.fc_var = nn.Linear(h_dim, latent_size)

        # for the gaussian likelihood
        self.log_sigma = nn.Parameter(torch.Tensor([0.0]))

        # to map z back to deecoder input
        self.z_rescale = nn.Linear(latent_size, h_dim)

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
        z = torch.rand(batch_size, latent_size)
        z = z.to(device)
        return self.decode(z)

    def gaussian_likelihood(self, x_hat, log_sigma, x):
        sigma = torch.exp(log_sigma)
        mean = x_hat
        dist = torch.distributions.Normal(mean, sigma)

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

        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_sigma = softclip(self.log_sigma, -6)
        
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, log_sigma, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo_loss = (kl - recon_loss)
        elbo_loss = elbo_loss.mean()

        return (elbo_loss, kl, recon_loss, x_hat)


#print(f'> Number of VAE parameters {len(torch.nn.utils.parameters_to_vector(VAE().parameters()))}')


# %% [markdown]
# ** Basic VAE **

# %%
from OldModels import CondenseEncoder, CondenseDecoder
print(image_channels)
vae_b_enc = CondenseEncoder(latent_size=latent_size, n_channels=image_channels)
vae_b_dec = CondenseDecoder(latent_size=latent_size, batch_size=batch_size, n_channels=image_channels)
V = VAE(vae_b_enc, vae_b_dec).to(device)
elo_loss, kl_loss, recon_loss = TrainModel(V, 0)
PlotAllLoss([elo_loss, kl_loss, recon_loss], ["EBLO", "KL", "Recon"])
PlotLoss(elo_loss)

# %% tags=[]
vae_res_enc = resnet18_encoder(False, False)
vae_res_dec = resnet18_decoder(
    latent_dim=latent_size,
    input_height=image_size,
    first_conv=False,
    maxpool1=False
)
Vres = VAE(vae_res_enc, vae_res_dec).to(device)
elo_loss, kl_loss, recon_loss = TrainModel(Vres, 1)
PlotAllLoss([elo_loss, kl_loss, recon_loss], ["EBLO", "KL", "Recon"])
PlotLoss(elo_loss)

# %%
PlotModelRandomGeneratedSample(Vres)

# %%
PlotSmallRandomSample(Vres)

# %%
PlotLatentSpace(Vres)

# %%
vae_res_enc2 = resnet18_encoder(False, False)
vae_res_dec2 = resnet18_decoder(
    latent_dim=latent_size,
    input_height=image_size,
    first_conv=False,
    maxpool1=False
)
Vres2 = VAE(vae_res_enc2, vae_res_dec2).to(device)
RestoreModel(Vres2, 'Vres-10hr')

# %%
PlotModelSampleEncoding(Vres2)

# %%
PlotSmallRandomSample(Vres2)
