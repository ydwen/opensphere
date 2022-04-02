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


def conv1x1(in_planes: int, out_planes: int, stride: int = 1,
            bias: bool = True) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=bias)

class ConvBlock(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super(ConvBlock, self).__init__()
        
        has_norm = norm_layer == nn.BatchNorm2d or norm_layer == nn.BatchNorm1d
        bias = not has_norm
        self.conv1 = conv3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super(BasicBlock, self).__init__()

        has_norm = norm_layer == nn.BatchNorm2d or norm_layer == nn.BatchNorm1d
        bias = not has_norm
        self.conv1 = conv3x3(inplanes, planes, bias=bias)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=bias)
        self.bn2 = norm_layer(planes)

        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)
        out = out + identity

        return out

class Bottleneck(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super(Bottleneck, self).__init__()
        width = int(planes // 2)
        has_norm = norm_layer == nn.BatchNorm2d or norm_layer == nn.BatchNorm1d
        bias = not has_norm

        self.conv1 = conv1x1(inplanes, width, bias=bias)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, bias=bias)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes, bias=bias)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class SFNet_deprecated(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        in_channel: Optional[int] = 3,
        channels: Optional[List[int]] = [64, 128, 256, 512],
        out_channel: Optional[int] = 512,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(SFNet_deprecated, self).__init__()

        norm_layer = nn.Identity
        self._norm_layer = norm_layer

        self.layer1 = self._make_layer(block, in_channel,
                                       channels[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, channels[0],
                                       channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[1],
                                       channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[2],
                                       channels[3], layers[3], stride=2)
        self.fc = nn.Linear(channels[3] * 7 * 7, out_channel)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)


    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], inplanes: int,
                    planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer

        layers = []
        layers.append(ConvBlock(inplanes, planes, stride, norm_layer=norm_layer))
        for _ in range(0, blocks):
            layers.append(block(planes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _sfnet_deprecated(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
) -> SFNet_deprecated:
    model = SFNet_deprecated(block, layers, **kwargs)

    return model


def sfnet4_deprecated(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SFNet_deprecated:
    r"""SFNet-4 model from
    `"SphereFace: Deep Hypersphere Embedding for Face Recognition" <https://arxiv.org/pdf/1704.08063.pdf>`_.
    """
    return _sfnet_deprecated('sfnet4_deprecated', BasicBlock, [0, 0, 0, 0], **kwargs)


def sfnet10_deprecated(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SFNet_deprecated:
    r"""SFNet-10 model from
    `"SphereFace: Deep Hypersphere Embedding for Face Recognition" <https://arxiv.org/pdf/1704.08063.pdf>`_.
    """
    return _sfnet_deprecated('sfnet10_deprecated', BasicBlock, [0, 1, 2, 0], **kwargs)


def sfnet20_deprecated(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SFNet_deprecated:
    r"""SFNet-20 model from
    `"SphereFace: Deep Hypersphere Embedding for Face Recognition" <https://arxiv.org/pdf/1704.08063.pdf>`_.
    """
    return _sfnet_deprecated('sfnet20_deprecated', BasicBlock, [1, 2, 4, 1], **kwargs)


def sfnet36_deprecated(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SFNet_deprecated:
    r"""SFNet-36 model from
    `"SphereFace: Deep Hypersphere Embedding for Face Recognition" <https://arxiv.org/pdf/1704.08063.pdf>`_.
    """
    return _sfnet_deprecated('sfnet36_deprecated', BasicBlock, [2, 4, 8, 2], **kwargs)


def sfnet64_deprecated(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SFNet_deprecated:
    r"""SFNet-64 model from
    `"SphereFace: Deep Hypersphere Embedding for Face Recognition" <https://arxiv.org/pdf/1704.08063.pdf>`_.
    """
    return _sfnet_deprecated('sfnet64_deprecated', BasicBlock, [3, 8, 16, 3], **kwargs)

