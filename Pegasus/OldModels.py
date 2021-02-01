import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

#######################################################################
#                               Basic AE                              #
#######################################################################

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

#######################################################################
#                           Basic VAE CNN                             #
#######################################################################

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

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

class CondenseEncoder(nn.Module):
    def __init__(self, f=16, n_channels=3,latent_size=32):
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
    def __init__(self, f=16, n_channels=3, latent_size=32, batch_size=256):
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


#######################################################################
#                              sigma-VAE                              #
#######################################################################

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

'''
from ResNetExample import resnet18_encoder, resnet18_decoder
vae_old_enc = resnet18_encoder(False, False)
vae_old_dec = resnet18_decoder(
    latent_dim=latent_size,
    input_height=image_size,
    first_conv=False,
    maxpool1=False
)
Vres_old = VAE(vae_old_enc, vae_old_dec).to(device)
RestoreModel(Vres_old, 'Vres-20hr')
'''

from ResNet import EncoderBlock, DecoderBlocker
class ResNetEncoder(nn.Module):

    def __init__(self, layers):
        super().__init__()

        self.current_planes = 64 # As per paper - we start with 64 planes

        # Layer 1
        self.conv1 = nn.Conv2d(3, self.current_planes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.current_planes)
        self.relu = nn.ReLU(inplace=True)

        # Layer 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_layer2 = self._make_layer(64, layers[0], stride=1)

        # Layer 3 -> 5
        self.conv_layer3 = self._make_layer(128, layers[1], stride=2)
        self.conv_layer4 = self._make_layer(256, layers[2], stride=2)
        self.conv_layer5 = self._make_layer(512, layers[3], stride=2)
        
        # Flatten planes into 1 number HxW
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, layer_planes, blocks, stride=None):
        downsample = None

        # Each layer spatially downsamples the previous layer's output
        # We don't want to do this on the first layer as that is done in a special conv
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.current_planes, layer_planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(layer_planes),
            )


        layers = []

        # If we downsample then the first of n block downsamples
        layers.append(EncoderBlock(self.current_planes, layer_planes, stride, downsample))
        self.current_planes = layer_planes

        # The rest don't touch the dimensions
        for _ in range(1, blocks):
            layers.append(EncoderBlock(layer_planes, layer_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.conv_layer2,
            self.conv_layer3,
            self.conv_layer4,
            self.conv_layer5,
            self.avgpool,
        )(x)
        
        x = torch.flatten(x, 1)
        return x


class ResNetDecoder(nn.Module):
    """
    Resnet in reverse order
    """

    def __init__(self, layers, latent_dim, input_height):
        super().__init__()

        self.current_planes = 512 # As per paper
        self.input_height = input_height

        # Linear layer to take us from latent to 4 x 4 x planes 
        self.linear = nn.Linear(latent_dim, self.current_planes * 4 * 4)
        
        # interpolate after linear layer using scale factor
        self.upscale1 = nn.Upsample(size=input_height // 32)

        # Opposite Layer 5 -> 3
        self.conv_layer5 = self._make_layer(256, layers[0], scale=2)
        self.conv_layer4 = self._make_layer(128, layers[1], scale=2)
        self.conv_layer3 = self._make_layer(64, layers[2], scale=2)

        # Reverse second layer 3x3 stride 2
        self.conv_layer2 = self._make_layer(64, layers[3], scale=2)
        
        # Reverse 7x7 stride 2 downsampling
        self.upscale = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.renorm = nn.Sigmoid()

    def _make_layer(self, target_planes, blocks, scale=1):
        upsample = None
        if scale != 1:
            upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale),
                nn.Conv2d(self.current_planes, target_planes , kernel_size=1, stride=1),
                nn.BatchNorm2d(target_planes),
            )

        layers = []
        layers.append(DecoderBlock(self.current_planes, target_planes, scale, upsample))
        self.current_planes = target_planes

        for _ in range(1, blocks):
            layers.append(DecoderBlock(self.current_planes, target_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 512, 4, 4)

        x = nn.Sequential(
            self.upscale1,
            self.conv_layer5,
            self.conv_layer4,
            self.conv_layer3,
            self.conv_layer2,
            self.upscale,
            self.conv1,
            self.renorm,
        )(x)
        return x


def ResNet18Encoder():
    return ResNetEncoder([2, 2, 2, 2])


def ResNet18Decoder(latent_dim, input_height):
    return ResNetDecoder([2, 2, 2, 2], latent_dim, input_height)

