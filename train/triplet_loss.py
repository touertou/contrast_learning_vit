import torch
import torch.nn as nn
import numpy as np
import os


def Triplet_loss(q, k, image_name):
    """
    计算三元组损失函数。

    参数:
        q: Tensor, 形状为 (1, 4096, 192) 的 anchor 向量。
        k: Tensor, 形状为 (1, 4096, 192) 的特征向量集合。
        image_name: str, 图像名称，用于加载对应的 positive 和 negative 索引文件。

    返回:
        loss: 三元组损失值 (标量 Tensor)。
    """
    # 移除第一个维度，变为 (4096, 192)
    q = q.squeeze(0)  # (4096, 192)
    k = k.squeeze(0)  # (4096, 192)

    # 加载 positive 和 negative 索引
    # anchor_index_path = os.path.join('../../data/npy_medsam/CT_Abd/label_gts2/anchor_index', image_name)
    # positive_index_path = os.path.join('../../data/npy_medsam/CT_Abd/label_gts/positive_index', image_name)
    # negative_index_path = os.path.join('../../data/npy_medsam/CT_Abd/label_gts/negative_index', image_name)
    #anchor_index_path = os.path.join('../../data/npy_medsam/CT_Abd/label_gts2/anchor_index', image_name)
    positive_index_path = os.path.join('../../data/data_synapse/npy/label_gts/positive_index', image_name)
    negative_index_path = os.path.join('../../data/data_synapse/npy/label_gts/negative_index', image_name)

    if not os.path.exists(positive_index_path) or not os.path.exists(negative_index_path):
        raise FileNotFoundError(f"Positive or negative index file not found for {image_name}")

    # 从 npy 文件中加载索引
    #anchor_index = np.load(anchor_index_path)
    positive_indices = np.load(positive_index_path)  # 形状为 (4096,)
    negative_indices = np.load(negative_index_path)  # 形状为 (4096,)

    # 根据索引挑选向量
    #anchor = q[anchor_index]
    positive = k[positive_indices]  # (4096, 192)
    negative = k[negative_indices]  # (4096, 192)

    # 创建三元组损失函数
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.3, p=2)

    # 计算损失
    #loss = triplet_loss_fn(anchor, positive, negative)
    loss = triplet_loss_fn(q, positive, negative)

    return loss

# q = torch.rand((1, 4096, 192), requires_grad=True)
# k = torch.rand((1, 4096, 192), requires_grad=True)
# triplet_loss = Triplet_loss(q, k, 'CT_Abd_0001-003.npy')
# print(triplet_loss)
