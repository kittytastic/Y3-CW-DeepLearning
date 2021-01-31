
#
# this code is based on https://github.com/PyTorchLightning/pytorch-lightning-bolts/pl_bolts/models/autoencoders/components.py, which is released under the Apache 2.0 licesne
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#
import torch
from torch import nn
from torch.nn import functional as F

class EncoderBlock(nn.Module):
    expansion = 1

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

        out += identity # The skip connection
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    # Mirror image of Encode Block - kind of; skip connection follows dataflow
    #           Conv1       Conv2
    # Encode    Shrink         -
    # Decode      -         Grow

    expansion = 1

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

        out += identity # The skip connection
        out = self.relu(out)
        return out



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


class FCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, depth):
        super().__init__()

        sf = out_dim/in_dim
        sf = sf**(1/float(depth))
        
        current_layers = in_dim
        layers=[]
        for i in range(depth-1):
            new_layers = int(current_layers*sf)
            layers.append(nn.Linear(current_layers, new_layers))
            current_layers = new_layers
        
        layers.append(nn.Linear(current_layers, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class FCCResNetEncoder(nn.Module):

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


class FCCResNetDecoder(nn.Module):
    """
    Resnet in reverse order
    """

    def __init__(self, layers, latent_dim, out_img_size, fc_depth):
        super().__init__()

        self.current_planes = 512 # As per paper

        self.starting_size = out_img_size // 32 # We scale up to 32x starting size
        self.fc_upscale = FCLayer(latent_dim, 512*self.starting_size*self.starting_size, fc_depth) # FC upscale to a size 

        # Opposite Layer 5 -> 3
        self.conv_layer5 = self._make_layer(256, layers[0], scale=2) # 256 x 2x x 2x
        self.conv_layer4 = self._make_layer(128, layers[1], scale=2) # 128 x 4x x 4x
        self.conv_layer3 = self._make_layer(64, layers[2], scale=2) # 64 x 8x x 8x

        # Reverse second layer 3x3 stride 2
        self.conv_layer2 = self._make_layer(64, layers[3], scale=2) # 32 x 16x x 16x
        
        # Reverse 7x7 stride 2 downsampling
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


        
def FCCResNet18Encoder(latent_size):
    return FCCResNetEncoder([2, 2, 2, 2], latent_size, 3)


def FCCResNet18Decoder(latent_size, img_size):
    return FCCResNetDecoder([2, 2, 2, 2], latent_size, img_size, 3)

