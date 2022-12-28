import torch
import numpy as np

def CostVolume(input_feature, candidate_feature, position="left", method="subtract", k=4, channel=32, D=256, batch_size=4):

    if position != "left" and position != "right":
        raise Exception('invalid cost volume direction')
    origin = input_feature
    candidate = candidate_feature
    cost_volume = torch.FloatTensor(input_feature.size()[0],
                             D // 2**k,
                             input_feature.size()[1],
                             input_feature.size()[2],
                             input_feature.size()[3]).zero_().cuda()
    # oMinusM_List = []
    if position == "left":
        for disparity in range(D // 2**k):
            if disparity > 0:
                cost_volume[:, disparity, :, :, disparity:] = origin[:, :, :, disparity:] - candidate[:, :, :, :-disparity]
            else:
                cost_volume[:, disparity, :, :, :] = origin - candidate

        return cost_volume
