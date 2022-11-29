"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import os
import os.path as osp

import copy
import math
from typing import Tuple

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.JDC.model import JDCNet
from Utils.ASR.models import ASRCNN

class DownSample(nn.Module):
    """DownSampling module
    """    
    def __init__(self, layer_type: str):
        """Constructor

        Args:
            layer_type (str): Specify the behaviour of the downsampling procedure.
        
        **Assert** layer_type must be a value in ["none","timepreserve","half"]
        """        
        super().__init__()
        assert(layer_type in ["none","timepreserve","half"])
        self.layer_type = layer_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method

        Args:
            x (torch.Tensor): _description_

        Raises:
            RuntimeError: _description_

        Returns:
            torch.Tensor: _description_
        """        
        if self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        if self.layer_type == 'half':
            return F.avg_pool2d(x, 2)
        return x

class UpSample(nn.Module):
    """UpSampling Module
    """    
    def __init__(self, layer_type: str):
        """Constructor

        Args:   
            layer_type (str): Specify the behaviour of the downsampling procedure.
        
        **Assert** layer_type must be a value in ["none","timepreserve","half"] 
        """        
        super().__init__()
        assert(layer_type in ["none","timepreserve","half"])
        self.layer_type = layer_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Raises:
            RuntimeError: _description_

        Returns:
            torch.Tensor: _description_
        """        
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    """Residual block definition
    """    
    def __init__(self, 
                 dim_in: int, 
                 dim_out: int, 
                 actv: torch.nn = nn.LeakyReLU(0.2),
                 normalize: bool = False, 
                 downsample: str = 'none'):
        """Constructor

        Args:
            dim_in (int): Number of input channels
            dim_out (int): Number of output channels
            actv (torch.nn, optional): Activation function. Defaults to nn.LeakyReLU(0.2).
            normalize (bool, optional): True-> Yes, False-> No. Defaults to False.
            downsample (str, optional): If downsampling is needed for this module. Defaults to 'none'. Values admitted ['timepreserve', 'half']
        """        
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample: DownSample = DownSample(downsample)
        self.learned_sc: bool = dim_in != dim_out # If the Nr. in-channels != Nr. out-channels, we need an interface
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in: int, dim_out: int):
        """ Method that initialize the component of the block

        Args:
            dim_in (int): Nr. of input channels
            dim_out (int): Nr. of output channels
        """
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True) # True: learnable affine parameters
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True) # True: learnable affine parameters
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False) # from dim_in to dim_out

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class AdaIN(nn.Module):
    """Adaptive Instance Module
    """    
    def __init__(self, style_dim: int, num_features: int):
        """_summary_

        Args:
            style_dim (int): N-features of style embedding
            num_features (int): N-features of the adaptive module
        """        
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False) # False: NOT learnable affine parameters
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, 
                 dim_in: int, 
                 dim_out: int, 
                 style_dim: int = 64, 
                 w_hpf: int = 0,
                 actv: torch.nn = nn.LeakyReLU(0.2), 
                 upsample: str = 'none'):
        """Constructor

        Args:
            dim_in (int): Number of input channels
            dim_out (int): Number of output channels
            style_dim (int): N-feature of style embedding
            w_hpf (int): Weight of high pass filter
            actv (torch.nn, optional): Activation function. Defaults to nn.LeakyReLU(0.2).
            upsampling (str, optional): If upsampling is needed for this module. Defaults to 'none'. Values admitted ['timepreserve', 'half']
        """
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = UpSample(upsample)
        self.learned_sc = dim_in != dim_out # If the Nr. in-channels != Nr. out-channels, we need an interface
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in: int, dim_out: int, style_dim: int = 64):
        """Initialize component of the module

        Args:
            dim_in (int): Number of input channels
            dim_out (int): Number of output channels
            style_dim (int): N-feature of style embedding
        """        
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc: 
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    """Generator Module
    """    
    def __init__(self, dim_in: int = 48, style_dim: int = 48, max_conv_dim: int = 48*8, w_hpf: int = 1, F0_channel: int = 0):
        """_summary_

        Args:
            dim_in (int, optional): N_Channel in input of the Generator (think at it as an interface from outside to generator). Defaults to 48.
            style_dim (int, optional): N-Features of Style Embedding. Defaults to 48.
            max_conv_dim (int, optional): Max number of channel for a convolutional block. Defaults to 48*8.
            w_hpf (int, optional): Weight of the High Pass Filter. Defaults to 1.
            F0_channel (int, optional): N-feature that F0 propose as output. Defaults to 0.
        """        
        
        super().__init__()
        self.stem = nn.Conv2d(1, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_out = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 1, 1, 1, 0))
        self.F0_channel = F0_channel
        
        # down/up-sampling blocks
        repeat_num = 4 #int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for lid in range(repeat_num):
            if lid in [1, 3]:
                _downtype = 'timepreserve'
            else:
                _downtype = 'half'

            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=_downtype))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                                w_hpf=w_hpf, upsample=_downtype))  # stack-like
            dim_in = dim_out

        # bottleneck blocks (encoder)
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
        
        # F0 blocks 
        if F0_channel != 0:
            self.decode.insert(
                0, AdainResBlk(dim_out + int(F0_channel / 2), dim_out, style_dim, w_hpf=w_hpf))
        
        # bottleneck blocks (decoder)
        for _ in range(2):
            self.decode.insert(
                    0, AdainResBlk(dim_out + int(F0_channel / 2), dim_out + int(F0_channel / 2), style_dim, w_hpf=w_hpf))
        
        if F0_channel != 0:
            self.F0_conv = nn.Sequential(
                ResBlk(F0_channel, int(F0_channel / 2), normalize=True, downsample="half"),
            )
        

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None, F0=None):            
        x = self.stem(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
            
        if F0 is not None:
            F0 = self.F0_conv(F0)
            F0 = F.adaptive_avg_pool2d(F0, [x.shape[-2], x.shape[-1]])
            x = torch.cat([x, F0], axis=1)

        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])

        return self.to_out(x)


class MappingNetwork(nn.Module):
    """Mapping network module
    """    
    def __init__(self, latent_dim: int = 16, style_dim: int = 48, num_domains: int = 2, hidden_dim: int = 384):
        """Constructor

        Args:
            latent_dim (int, optional): _description_. Defaults to 16.
            style_dim (int, optional): _description_. Defaults to 48.
            num_domains (int, optional): _description_. Defaults to 2.
            hidden_dim (int, optional): _description_. Defaults to 384.
        """        
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, hidden_dim)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    """Style encoder module
    """    
    def __init__(self, dim_in: int = 48, style_dim: int = 48, num_domains: int = 2, max_conv_dim: int = 384):
        """Constructor

        Args:
            dim_in (int, optional): N_Channel in input of the StyleEncoder (think at it as an interface from outside to stylencoder). Defaults to 48.
            style_dim (int, optional): N-features of the style embedding. Defaults to 48.
            num_domains (int, optional): Number of domains that the style encoder has to take into account. Defaults to 2.
            max_conv_dim (int, optional): Max number of convolutional features map. Defaults to 384.
        """        
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)

        h = h.view(h.size(0), -1)
        out = []

        for layer in self.unshared:
            out += [layer(h)]

        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s

class Discriminator(nn.Module):
    def __init__(self, dim_in=48, num_domains=2, max_conv_dim=384, repeat_num=4):
        super().__init__()
        
        # real/fake discriminator
        self.dis = Discriminator2d(dim_in=dim_in, num_domains=num_domains,
                                    max_conv_dim=max_conv_dim, repeat_num=repeat_num)
        # adversarial classifier
        self.cls = Discriminator2d(dim_in=dim_in, num_domains=num_domains,
                                    max_conv_dim=max_conv_dim, repeat_num=repeat_num)                             
        self.num_domains = num_domains
        
    def forward(self, x, y):
        return self.dis(x, y)

    def classifier(self, x):
        return self.cls.get_feature(x)


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Discriminator2d(nn.Module):
    def __init__(self, dim_in=48, num_domains=2, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out

    def forward(self, x, y):
        out = self.get_feature(x)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


def build_model(args: Munch, F0_model: JDCNet, ASR_model: ASRCNN) -> Tuple[Munch,Munch]:
    """Methon that given Model properties and submodule, produce the model net

    Args:
        args (Munch): Net paramenter
        F0_model (JDCNet): JDCNet Model
        ASR_model (ASRCNN): ASRCNN Model

    Returns:
        (Munch, Munch): Tuple containing 2 Model Copy, one for training, one for inference.
    """    
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    discriminator = Discriminator(args.dim_in, args.num_domains, args.max_conv_dim, args.n_repeat)
    generator_ema: Generator = copy.deepcopy(generator)
    mapping_network_ema: MappingNetwork = copy.deepcopy(mapping_network)
    style_encoder_ema: StyleEncoder = copy.deepcopy(style_encoder)
        
    nets = Munch(generator=generator,
                mapping_network=mapping_network,
                style_encoder=style_encoder,
                discriminator=discriminator,
                f0_model=F0_model,
                asr_model=ASR_model)
    
    nets_ema = Munch(generator=generator_ema,
                        mapping_network=mapping_network_ema,
                        style_encoder=style_encoder_ema)

    return nets, nets_ema