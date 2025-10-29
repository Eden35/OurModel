import torch
import numpy as np
import torch.nn.functional as F

def rescale_intensity(data, new_min=0, new_max=1, group=4, eps=1e-20):
    '''
    rescale pytorch batch data
    :param data: N*1*H*W
    :return: data with intensity ranging from 0 to 1
    '''
    bs, c, h, w = data.size(0), data.size(1), data.size(2), data.size(3)
    data = data.view(bs * c, -1)
    old_max = torch.max(data, dim=1, keepdim=True).values
    old_min = torch.min(data, dim=1, keepdim=True).values

    new_data = (data - old_min + eps) / (old_max - old_min + eps) * (new_max - new_min) + new_min
    new_data = new_data.view(bs, c, h, w)
    return new_data

# def rescale_intensity(data, new_min=0, new_max=1, eps=1e-20):
#     bs, c, h, w = data.size(0), data.size(1), data.size(2), data.size(3)
#     data_flat = data.view(bs * c, -1)  # [bs*c, H*W]
#     old_max = torch.max(data_flat, dim=1, keepdim=True).values  # [bs*c, 1]
#     old_min = torch.min(data_flat, dim=1, keepdim=True).values  # [bs*c, 1]
#
#     # 创建与 data_flat 同形状的 mask
#     mask = (old_max - old_min) < eps
#     mask = mask.expand_as(data_flat)  # 关键修复：将 mask 扩展到 [bs*c, H*W]
#
#     new_data = torch.zeros_like(data_flat)
#     # 处理非全同值通道
#     new_data[~mask] = (data_flat[~mask] - old_min.expand_as(data_flat)[~mask] + eps) / \
#                       (old_max.expand_as(data_flat)[~mask] - old_min.expand_as(data_flat)[~mask] + eps) * \
#                       (new_max - new_min) + new_min
#     # 处理全同值通道
#     new_data[mask] = new_min
#
#     return new_data.view(bs, c, h, w)