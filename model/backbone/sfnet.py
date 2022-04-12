import torch
import torch.nn as nn

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['SFNet', 'sfnet4', 'sfnet10',
           'sfnet20', 'sfnet36', 'sfnet64']

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

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

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


class SFNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        in_channel: Optional[int] = 3,
        channels: Optional[List[int]] = [64, 128, 256, 512],
        out_channel: Optional[int] = 512,
        dropout: Optional[float] = 0.,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(SFNet, self).__init__()
        if not norm_layer:
            norm_layer = nn.Identity
            self.features = nn.Identity()
        else:
            norm_layer = nn.BatchNorm2d
            self.features = nn.BatchNorm1d(out_channel, eps=1e-05)
            nn.init.constant_(self.features.weight, 1.0)

        self._norm_layer = norm_layer

        self.layer1 = self._make_layer(block, in_channel,
                                       channels[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, channels[0],
                                       channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[1],
                                       channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[2],
                                       channels[3], layers[3], stride=2)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(channels[3] * 7 * 7, out_channel)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

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
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _sfnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> SFNet:
    model = SFNet(block, layers, **kwargs)
    if pretrained:
        pass
        #state_dict = load_state_dict_from_url(model_urls[arch],
        #                                      progress=progress)
        #model.load_state_dict(state_dict)
    return model


def sfnet4(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SFNet:
    r"""SFNet-4 model from
    `"SphereFace: Deep Hypersphere Embedding for Face Recognition" <https://arxiv.org/pdf/1704.08063.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _sfnet('sfnet4', BasicBlock, [0, 0, 0, 0], pretrained, progress,
                   **kwargs)


def sfnet10(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SFNet:
    r"""SFNet-10 model from
    `"SphereFace: Deep Hypersphere Embedding for Face Recognition" <https://arxiv.org/pdf/1704.08063.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _sfnet('sfnet10', BasicBlock, [0, 1, 2, 0], pretrained, progress,
                   **kwargs)


def sfnet20(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SFNet:
    r"""SFNet-20 model from
    `"SphereFace: Deep Hypersphere Embedding for Face Recognition" <https://arxiv.org/pdf/1704.08063.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _sfnet('sfnet20', BasicBlock, [1, 2, 4, 1], pretrained, progress,
                   **kwargs)


def sfnet36(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SFNet:
    r"""SFNet-36 model from
    `"SphereFace: Deep Hypersphere Embedding for Face Recognition" <https://arxiv.org/pdf/1704.08063.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _sfnet('sfnet36', BasicBlock, [2, 4, 8, 2], pretrained, progress,
                   **kwargs)


def sfnet64(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SFNet:
    r"""SFNet-64 model from
    `"SphereFace: Deep Hypersphere Embedding for Face Recognition" <https://arxiv.org/pdf/1704.08063.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _sfnet('sfnet64', BasicBlock, [3, 8, 16, 3], pretrained, progress,
                   **kwargs)

