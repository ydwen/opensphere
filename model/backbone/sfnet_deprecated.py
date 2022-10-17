import torch
import torch.nn as nn

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['SFNet_deprecated', 'sfnet4_deprecated',
           'sfnet10_deprecated', 'sfnet20_deprecated',
           'sfnet36_deprecated', 'sfnet64_deprecated']

def conv3x3(in_planes: int, out_planes: int, stride: int = 1,
            bias: bool = True) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=bias)

class ConvBlock(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int,
    ) -> None:
        super(ConvBlock, self).__init__()
        
        self.conv = conv3x3(inplanes, planes, stride, bias=True)
        self.act = nn.ReLU(inplace=True)

        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:

        out = self.conv(x)
        out = self.act(out)

        return out

class BasicBlock(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
    ) -> None:
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.act2 = nn.ReLU(inplace=True)

        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)

    def forward(self, x: Tensor) -> Tensor:

        identity = x

        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)

        out = out + identity

        return out


class SFNet_deprecated(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock]],
        layers: List[int],
        in_channel: Optional[int] = 3,
        channels: Optional[List[int]] = [64, 128, 256, 512],
        out_channel: Optional[int] = 512,
    ) -> None:
        super(SFNet_deprecated, self).__init__()

        self.layer1 = self._make_layer(block, in_channel,
                                       channels[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, channels[0],
                                       channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[1],
                                       channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[2],
                                       channels[3], layers[3], stride=2)
        self.fc = nn.Linear(channels[3] * 7 * 7, out_channel)

    def _make_layer(self, block: Type[Union[BasicBlock]], inplanes: int,
                    planes: int, blocks: int, stride: int = 1) -> nn.Sequential:

        layers = [ConvBlock(inplanes, planes, stride)]
        for _ in range(0, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def sfnet4_deprecated(**kwargs: Any) -> SFNet_deprecated:
    """SFNet-4 model from
    `SphereFace: Deep Hypersphere Embedding for Face Recognition`
    (https://arxiv.org/pdf/1704.08063.pdf)
    """
    return SFNet_deprecated(BasicBlock, [0, 0, 0, 0], **kwargs)

def sfnet10_deprecated(**kwargs: Any) -> SFNet_deprecated:
    """SFNet-10 model from
    `SphereFace: Deep Hypersphere Embedding for Face Recognition`
    (https://arxiv.org/pdf/1704.08063.pdf)
    """
    return SFNet_deprecated(BasicBlock, [0, 1, 2, 0], **kwargs)

def sfnet20_deprecated(**kwargs: Any) -> SFNet_deprecated:
    """SFNet-20 model from
    `SphereFace: Deep Hypersphere Embedding for Face Recognition`
    (https://arxiv.org/pdf/1704.08063.pdf)
    """
    return SFNet_deprecated(BasicBlock, [1, 2, 4, 1], **kwargs)

def sfnet36_deprecated(**kwargs: Any) -> SFNet_deprecated:
    """SFNet-36 model from
    `SphereFace: Deep Hypersphere Embedding for Face Recognition`
    (https://arxiv.org/pdf/1704.08063.pdf)
    """
    return SFNet_deprecated(BasicBlock, [2, 4, 8, 2], **kwargs)


def sfnet64_deprecated(**kwargs: Any) -> SFNet_deprecated:
    """SFNet-64 model from
    `SphereFace: Deep Hypersphere Embedding for Face Recognition`
    (https://arxiv.org/pdf/1704.08063.pdf)
    """
    return SFNet_deprecated(BasicBlock, [3, 8, 16, 3], **kwargs)
