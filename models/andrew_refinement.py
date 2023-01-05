import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.cost_volume import CostVolume
from collections import OrderedDict

#StereoNet CostVolume Refinement
class Refinement(torch.nn.Module):
    """
    Several of these classes will be instantiated to perform the *cascading* refinement.  Refer to the original paper for a full discussion.
    """

    def __init__(self) -> None:
        super().__init__()

        dilations = [1, 2, 4, 8, 1, 1]

        net: OrderedDict[str, nn.Module] = OrderedDict()

        net['segment_0_conv_0'] = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)

        for block_idx, dilation in enumerate(dilations):
            net[f'segment_1_res_{block_idx}'] = ResBlock(in_channels=32, out_channels=32, kernel_size=3, padding=dilation, dilation=dilation)

        net['segment_2_conv_0'] = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

        self.net = nn.Sequential(net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name, missing-function-docstring
        x = self.net(x)
        return x

class ResBlock(torch.nn.Module):
    """
    Just a note, in the original paper, there is no discussion about padding; however, both the ZhiXuanLi and the X-StereoLab implementation using padding.
    This does make sense to maintain the image size after the feature extraction has occured.

    X-StereoLab uses a simple Res unit with a single conv and summation while ZhiXuanLi uses the original residual unit implementation.
    This class also uses the original implementation with 2 layers of convolutions.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels)
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)

        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels)
        self.activation_2 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
        """
        Original Residual Unit: https://arxiv.org/pdf/1603.05027.pdf (Fig 1. Left)
        """

        res = self.conv_1(x)
        res = self.batch_norm_1(res)
        res = self.activation_1(res)
        res = self.conv_2(res)
        res = self.batch_norm_2(res)

        # I'm not really sure why the type definition is required here... nn.Conv2d already returns type Tensor...
        # So res should be of type torch.Tensor AND x is already defined as type torch.Tensor.
        out: torch.Tensor = res + x
        out = self.activation_2(out)

        return out