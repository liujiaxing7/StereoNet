import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.cost_volume import CostVolume

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class AffinityFeature(nn.Module):
    def __init__(self, win_h, win_w, dilation, cut):
        super(AffinityFeature, self).__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.dilation = dilation
        self.cut = 0

    def padding(self, x, win_h, win_w, dilation):
        pad_t = (win_w // 2 * dilation, win_w // 2 * dilation,
                 win_h // 2 * dilation, win_h // 2 * dilation)
        out = F.pad(x, pad_t, mode='constant')
        return out

    def forward(self, feature):
        B, C, H, W = feature.size()
        feature = F.normalize(feature, dim=1, p=2)

        # affinity = []
        # pad_feature = self.padding(feature, win_w=self.win_w, win_h=self.win_h, dilation=self.dilation)
        # for i in range(self.win_w):
        #     for j in range(self.win_h):
        #         if (i == self.win_w // 2) & (j == self.win_h // 2):
        #             continue
        #         simi = self.cal_similarity(
        #             pad_feature[:, :, self.dilation*j:self.dilation*j+H, self.dilation*i:self.dilation*i+W],
        #             feature, self.simi_type)
        #
        #         affinity.append(simi)
        # affinity = torch.stack(affinity, dim=1)
        #
        # affinity[affinity < self.cut] = self.cut

        unfold_feature = nn.Unfold(
            kernel_size=(self.win_h, self.win_w), dilation=self.dilation, padding=self.dilation)(feature)
        all_neighbor = unfold_feature.reshape(B, C, -1, H, W).transpose(1, 2)
        num = (self.win_h * self.win_w) // 2
        neighbor = torch.cat((all_neighbor[:, :num], all_neighbor[:, num+1:]), dim=1)
        feature = feature.unsqueeze(1)
        affinity = torch.sum(neighbor * feature, dim=2)
        # affinity[affinity < self.cut] = self.cut
        self.cut=torch.tensor(self.cut, dtype=torch.float).to("cuda")
        affinity=torch.where(affinity < self.cut,self.cut,affinity )

        return affinity

class StructureFeature(nn.Module):
    def __init__(self, affinity_settings, sfc):
        super(StructureFeature, self).__init__()

        self.win_w = affinity_settings['win_w']
        self.win_h = affinity_settings['win_h']
        self.dilation = affinity_settings['dilation']

        self.sfc = sfc

        in_c = self.win_w * self.win_h - 1

        self.sfc_conv1 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv2 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv3 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv4 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))

    def forward(self, x):

        affinity1 = AffinityFeature(self.win_h, self.win_w, self.dilation[0], 0)(x)
        affinity2 = AffinityFeature(self.win_h, self.win_w, self.dilation[1], 0)(x)
        affinity3 = AffinityFeature(self.win_h, self.win_w, self.dilation[2], 0)(x)
        affinity4 = AffinityFeature(self.win_h, self.win_w, self.dilation[3], 0)(x)

        affi_feature1 = self.sfc_conv1(affinity1)
        affi_feature2 = self.sfc_conv2(affinity2)
        affi_feature3 = self.sfc_conv3(affinity3)
        affi_feature4 = self.sfc_conv4(affinity4)

        out_feature = torch.cat((affi_feature1, affi_feature2, affi_feature3, affi_feature4), dim=1)
        affinity = torch.cat((affinity1, affinity2, affinity3, affinity4), dim=1)

        # out_feature = torch.cat((affi_feature1, affi_feature2, affi_feature3), dim=1)
        # affinity = torch.cat((affinity1, affinity2, affinity3), dim=1)

        return out_feature, affinity
        # return affinity1, affinity1

class feature_extraction(nn.Module):
    def __init__(self, structure_fc, fuse_mode, affinity_settings):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.sfc = structure_fc
        self.fuse_mode = fuse_mode
        self.win_w = affinity_settings['win_w']
        self.win_h = affinity_settings['win_h']
        self.dilation = affinity_settings['dilation']

        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

        if self.sfc > 0:
            if fuse_mode == 'aggregate':
                self.embedding = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(128, 128, kernel_size=1, padding=0, stride=1, bias=False))

                in_c = self.win_w * self.win_h - 1
                self.sfc_conv1 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                               nn.ReLU(inplace=True))
                self.sfc_conv2 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                               nn.ReLU(inplace=True))
                self.sfc_conv3 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                               nn.ReLU(inplace=True))
                self.sfc_conv4 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                               nn.ReLU(inplace=True))

                self.lastconv = nn.Sequential(convbn(4*self.sfc, 32, 3, 1, 1, 1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False))

                # self.lastconv = nn.Sequential(convbn(320 + 4*self.sfc, 128, 3, 1, 1, 1),
                #                               nn.ReLU(inplace=True),
                #                               nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False))

                self.to_sf = StructureFeature(affinity_settings, self.sfc)

            elif fuse_mode == 'separate':
                # self.embedding_l1 = nn.Sequential(convbn(32, 64, kernel_size=3, stride=1, pad=1, dilation=1),
                #                                   nn.ReLU(inplace=True),
                #                                   nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False))
                # self.to_sf_l1 = StructureFeature(affinity_settings, self.sfc)

                self.embedding_l2 = nn.Sequential(convbn(64, 64, kernel_size=3, stride=1, pad=1, dilation=1),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False))
                self.to_sf_l2 = StructureFeature(affinity_settings, self.sfc)

                self.embedding_l3 = nn.Sequential(convbn(128, 64, kernel_size=3, stride=1, pad=1, dilation=1),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False))
                self.to_sf_l3 = StructureFeature(affinity_settings, self.sfc)

                self.embedding_l4 = nn.Sequential(convbn(128, 64, kernel_size=3, stride=1, pad=1, dilation=1),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False))
                self.to_sf_l4 = StructureFeature(affinity_settings, self.sfc)

                # self.lastconv = nn.Sequential(convbn(3 * 4 * self.sfc, 32, 3, 1, 1, 1),
                #                               nn.ReLU(inplace=True),
                #                               nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False))

                self.lastconv = nn.Sequential(convbn(320 + 3 * 4 * self.sfc, 128, 3, 1, 1, 1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output_l1 = self.layer1(output)
        output_l2 = self.layer2(output_l1)
        output_l3 = self.layer3(output_l2)
        output_l4 = self.layer4(output_l3)

        # output_l1 = F.interpolate(output_l1, (output_l4.size()[2], output_l4.size()[3]),
        #                           mode='bilinear', align_corners=True)

        cat_feature = torch.cat((output_l2, output_l3, output_l4), 1)

        if self.sfc > 0:
            if self.fuse_mode == 'aggregate':
                embedding = self.embedding(cat_feature)

                cat_sf, affinity = self.to_sf(embedding)

                # output_feature = self.lastconv(torch.cat((cat_feature, cat_sf), dim=1))
                output_feature = self.lastconv(cat_sf)

            elif self.fuse_mode == 'separate':
                # embedding_l1 = self.embedding_l1(output_l1)
                # l1_sf, l1_affi = self.to_sf_l1(embedding_l1)

                embedding_l2 = self.embedding_l2(output_l2.detach())
                l2_sf, l2_affi = self.to_sf_l2(embedding_l2)

                embedding_l3 = self.embedding_l3(output_l3.detach())
                l3_sf, l3_affi = self.to_sf_l3(embedding_l3)

                embedding_l4 = self.embedding_l4(output_l4.detach())
                l4_sf, l4_affi = self.to_sf_l4(embedding_l4)

                # output_feature = self.lastconv(torch.cat((l2_sf, l3_sf, l4_sf), dim=1))
                output_feature = self.lastconv(torch.cat((cat_feature, l2_sf, l3_sf, l4_sf), dim=1))
                affinity = torch.cat((l2_affi, l3_affi, l4_affi), dim=1)

            return output_feature

        else:
            output_feature = self.lastconv(cat_feature)

            return output_feature

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes, affine=False)
            self.norm2 = nn.InstanceNorm2d(planes, affine=False)
            self.norm3 = nn.InstanceNorm2d(planes, affine=False)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        x = self.downsample(x)

        return self.relu(x + y)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64, affine=False)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=1)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, x.shape[0] // 2, dim=0)

        return x

class StereoNet(nn.Module):
    def __init__(self, batch_size, cost_volume_method):
        super(StereoNet, self).__init__()

        self.batch_size = batch_size
        self.cost_volume_method = cost_volume_method
        cost_volume_channel = 32
        if cost_volume_method == "subtract":
            cost_volume_channel = 32
        elif cost_volume_method == "concat":
            cost_volume_channel = 64
        else:
            print("cost_volume_method is not right")

        affinity_settings = {}
        affinity_settings['win_w'] = 3
        affinity_settings['win_h'] = 3
        affinity_settings['dilation'] = [1, 2, 4, 8]

        self.feature_extraction = feature_extraction(0, 'separate', affinity_settings)

        # self.fnet = BasicEncoder(output_dim=32, norm_fn='instance', dropout=0)

        self.downsampling = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.Conv2d(32, 32, 5, stride=2, padding=2),
            nn.Conv2d(32, 32, 5, stride=2, padding=2),
            nn.Conv2d(32, 32, 5, stride=2, padding=2),
        )

        self.res = nn.Sequential(
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            nn.Conv2d(32, 32, 3, 1, 1),
        )

        """ using 3d conv to instead the Euclidean distance"""
        self.cost_volume_filter = nn.Sequential(
            MetricBlock(cost_volume_channel, 32),
            MetricBlock(32, 32),
            MetricBlock(32, 32),
            MetricBlock(32, 32),
            nn.Conv3d(32, 1, 3, padding=1),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            ResBlock(32, 32, dilation=1),
            ResBlock(32, 32, dilation=2),
            ResBlock(32, 32, dilation=4),
            ResBlock(32, 32, dilation=8),
            ResBlock(32, 32, dilation=1),
            ResBlock(32, 32, dilation=1),
            nn.Conv2d(32, 1, 3, padding=1),
            # nn.ReLU(),
        )

    def forward_once_1(self, x):
        output = self.downsampling(x)

        output = self.res(output)

        return output

    def forward_stage1(self, input_l, input_r):
        output_l = self.feature_extraction(input_l)
        output_r = self.feature_extraction(input_r)

        return output_l, output_r

    def forward_once_2(self, cost_volume):
        """the index cost volume's dimension is not right for conv3d here, so we change it"""
        cost_volume = cost_volume.permute([0, 2, 1, 3, 4])

        output = self.cost_volume_filter(cost_volume)  # [batch_size, channel, disparity, h, w]
        # output = cost_volume[:, 0:1, :, :, :]
        disparity_low = output

        return disparity_low  # low resolution disparity map

    def forward_stage2(self, feature_l, feature_r):
        cost_v_l = CostVolume(feature_l, feature_r, "left", method=self.cost_volume_method, k=4, batch_size=self.batch_size)

        disparity_low = self.forward_once_2(cost_v_l)
        disparity_low = torch.squeeze(disparity_low, dim=1)

        return disparity_low

    def forward_stage3(self, disparity_low, left):
        """upsample and concatenate"""
        d_high = nn.functional.interpolate(disparity_low, [left.shape[2], left.shape[3]], mode='bilinear', align_corners=True)
        d_high = soft_argmin(d_high)

        d_concat = torch.cat([d_high, left], dim=1)

        d_refined = self.refine(d_concat)

        return d_refined

    def forward(self, left, right):
        left_feature, right_feature = self.forward_stage1(left, right)
        disparity_low_l = self.forward_stage2(left_feature, right_feature)

        d_initial_l = nn.functional.interpolate(disparity_low_l, [left.shape[2], left.shape[3]], mode='bilinear',
                                                align_corners=True)
        d_initial_l = soft_argmin(d_initial_l)
        d_refined_l = self.forward_stage3(disparity_low_l, left)
        d_final_l = d_initial_l + d_refined_l
        # d_final_l = soft_argmin(d_initial_l + d_refined_l)

        d_final_l = nn.ReLU()(d_final_l)

        return d_final_l

class MetricBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1):
        super(MetricBlock, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channel, out_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv3d_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dilation=1, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        # To keep the shape of input and output same when dilation conv, we should compute the padding:
        # Reference:
        #   https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
        # padding = [(o-1)*s+k+(k-1)*(d-1)-i]/2, here the i is input size, and o is output size.
        # set o = i, then padding = [i*(s-1)+k+(k-1)*(d-1)]/2 = [k+(k-1)*(d-1)]/2      , stride always equals 1
        # if dilation != 1:
        #     padding = (3+(3-1)*(dilation-1))/2
        padding = dilation

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.p = padding
        self.d = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        out = self.relu2(out)

        return out

def soft_argmin(cost_volume):
    """Remove single-dimensional entries from the shape of an array."""
    # cost_volume_D_squeeze = torch.squeeze(cost_volume, dim=1)

    softmax = nn.Softmax(dim=1)
    disparity_softmax = softmax(cost_volume * -1)

    d_grid = torch.arange(cost_volume.shape[1], dtype=torch.float)
    d_grid = d_grid.reshape(-1, 1, 1)
    d_grid = d_grid.repeat((cost_volume.shape[0], 1, cost_volume.shape[2], cost_volume.shape[3])) # [batchSize, 1, h, w]
    d_grid = d_grid.to('cuda')

    tmp = disparity_softmax*d_grid
    arg_soft_min = torch.sum(tmp, dim=1, keepdim=True)

    return arg_soft_min
