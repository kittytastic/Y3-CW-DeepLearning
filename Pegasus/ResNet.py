
#
# this code is based on https://github.com/PyTorchLightning/pytorch-lightning-bolts/pl_bolts/models/autoencoders/components.py, which is released under the Apache 2.0 License
# which in turn used portions of code from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py which is released under BSD 3-Clause License
#
# Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
#

import torch
from torch import nn
#from torch.nn import functional as F


# Simple 2 conv block
class EncoderBlock(nn.Module):
    def __init__(self, in_planes, working_planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, working_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(working_planes)
        
        self.conv2 = nn.Conv2d(working_planes, working_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(working_planes)
        
        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2,
        )(x)

        out += identity # The magic skip connection
        out = self.relu(out)
        return out


# Simple 2 conv block
class DecoderBlock(nn.Module):
    # Encoder block in reverse
    #           Conv1       Conv2
    # Encode    Shrink         -
    # Decode      -         Grow

    def __init__(self, working_planes, out_planes, scale=1, upsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(working_planes, working_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(working_planes)
        
        self.conv2 = nn.Sequential(
                nn.Upsample(scale_factor=scale), 
                nn.Conv2d(working_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
            )
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        self.upsample = upsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        if self.upsample is not None:
            identity = self.upsample(x)
        
        out = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2,
        )(x)

        out += identity # The magic skip connection
        out = self.relu(out)
        return out

#
# Fully connected layer
# Take "in_dim" input, gives "out_dim" outputs
# Uses "depth" fully connected layers
# Intermediate layers are logarithmically scaled
#
class FCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, depth):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        sf = out_dim/in_dim
        sf = sf**(1/float(depth))
        
        current_layers = in_dim
        layers=[]
        for i in range(depth-1):
            new_layers = int(current_layers*sf)
            layers.append(nn.Linear(current_layers, new_layers))
            layers.append(self.relu)
            current_layers = new_layers
        
        layers.append(nn.Linear(current_layers, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Resnet encoder
# However use a fully connected layer at the end to take us from conv space to latent space
class FCResNetEncoder(nn.Module):

    def __init__(self, layers, latent_size, fc_depth):
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
        
        # Pool:  planes x H x W into: planes x 1 x 1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer to latent space
        self.fc_to_latent = FCLayer(512, latent_size, fc_depth)

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
        
        x = x.squeeze()
        x = self.fc_to_latent(x)
        return x


# Resnet decoder
# Does what the decoder does but backwards
# Uses a final 3x3 kernel instead of 7x7 to help recreate finer details
# Uses a fully connected layer at the begining to take us from latent space to conv space
class FCResNetDecoder(nn.Module):

    def __init__(self, layers, latent_dim, out_img_size, fc_depth):
        super().__init__()

        self.current_planes = 512 # As per paper

        self.starting_size = out_img_size // 32 # We scale up to 32x starting size
        self.fc_upscale = FCLayer(latent_dim, 512*self.starting_size*self.starting_size, fc_depth) # FC upscale to a size
        self.bn1 = nn.BatchNorm1d(512*self.starting_size*self.starting_size) 

        # Opposite Layer 5 -> 3
        self.conv_layer5 = self._make_layer(256, layers[0], scale=2) # 256 x 2x x 2x
        self.conv_layer4 = self._make_layer(128, layers[1], scale=2) # 128 x 4x x 4x
        self.conv_layer3 = self._make_layer(64, layers[2], scale=2) # 64 x 8x x 8x

        # Reverse second layer 3x3 stride 2
        self.conv_layer2 = self._make_layer(64, layers[3], scale=2) # 32 x 16x x 16x
        
        # Reverse 7x7 stride 2 downsampling, although use 3x3 kernel to help with finer details
        self.upscale = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True) # 16 x 32x x 32x 

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
        x = self.fc_upscale(x)
        x = self.bn1(x)
        x = x.view(x.size(0), 512, self.starting_size, self.starting_size)
    
        x = nn.Sequential(
            self.conv_layer5,
            self.conv_layer4,
            self.conv_layer3,
            self.conv_layer2,
            self.upscale,
            self.conv1,
            self.renorm,
        )(x)
        return x


# Uses the 2,2,2,2 config as proposed in paper
# There may be better distributions for layers
# but without an abundance of time I will assume Zhang et.al
# found 2,2,2,2 to be particually good for the number of parameters    
def FCResNet18Encoder(latent_size):
    return FCResNetEncoder([2, 2, 2, 2], latent_size, 3)


def FCResNet18Decoder(latent_size, img_size):
    return FCResNetDecoder([2, 2, 2, 2], latent_size, img_size, 3)