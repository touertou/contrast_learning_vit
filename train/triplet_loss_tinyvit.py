import torch
import torch.nn as nn
def Triplet_loss_tinyvit(q, k, positive_indices, negative_indices):
    """
    计算三元组损失函数。

    参数:
        q: Tensor, 形状为 (1, 4096, 192) 的 anchor 向量。
        k: Tensor, 形状为 (1, 4096, 192) 的特征向量集合。
        positive_indices: 正样本索引
        negative_indices: 负样本索引

    返回:
        loss: 三元组损失值 (标量 Tensor)。
    """
    # 移除第一个维度，变为 (4096, 192)
    q = q.squeeze(0)  # (4096, 192)
    k = k.squeeze(0)  # (4096, 192)

    # 根据索引挑选向量
    positive = k[positive_indices]  # (4096, 192)
    negative = k[negative_indices]  # (4096, 192)

    # 创建三元组损失函数
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.3, p=2)

    # 计算损失
    loss = triplet_loss_fn(q, positive, negative)

    return loss

