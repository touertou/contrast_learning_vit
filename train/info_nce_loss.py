import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss的实现，使用预生成的正负样本索引
    
    数学公式：
    L = -log(exp(q·k+/τ) / Σ exp(q·k/τ))
    
    其中：
    - q 是查询向量
    - k+ 是正样本向量
    - k 是所有样本向量
    - τ 是温度参数
    """
    def __init__(self, temperature=0.07):
        """
        参数:
            temperature: float, 温度参数，控制分布的平滑程度
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, q, k, positive_indices, negative_indices):
        """
        计算InfoNCE loss
        
        参数:
            features: Tensor, 形状为 (B, N, C) 的特征向量
            positive_indices: Tensor, 形状为 (B, N) 的正样本索引
            negative_indices: list, 包含负样本索引的列表
            
        返回:
            loss: InfoNCE loss值 (标量 Tensor)
        """
        batch_size, num_patches, feat_dim = q.shape
        device = q.device
        
        # 归一化特征向量
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # 初始化损失
        total_loss = 0
        valid_patches = 0
        
        for b in range(batch_size):
            for p in range(num_patches):
                # 获取当前patch的特征
                query = q[b, p]  # (C,)
                
                # 获取正样本索引
                pos_idx = positive_indices[b, p]
                if pos_idx == -1:  # 跳过无效的正样本
                    continue
                    
                # 获取正样本特征
                positive = k[b, pos_idx]  # (C,)
                
                # 获取负样本索引
                neg_indices = negative_indices[b][p]
                if isinstance(neg_indices, int) and neg_indices == -1:  # 跳过无效的负样本
                    continue
                    
                # 获取负样本特征
                negatives = k[b, neg_indices]  # (M, C)
                
                # 计算与正样本的相似度
                pos_sim = torch.sum(query * positive) / self.temperature
                
                # 计算与负样本的相似度
                neg_sim = torch.matmul(query, negatives.t()) / self.temperature
                
                # 拼接正负样本的相似度
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
                
                # 创建标签：第一个样本为正样本
                labels = torch.zeros(1, dtype=torch.long, device=device)
                
                # 计算交叉熵损失
                loss = F.cross_entropy(logits.unsqueeze(0), labels)
                
                total_loss += loss
                valid_patches += 1
        
        # 计算平均损失
        if valid_patches > 0:
            return total_loss / valid_patches
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


def InfoNCELossPatch(q, k, image_name):
    """
    计算InfoNCE loss，使用预先生成的patch索引。

    参数:
        q: Tensor, 形状为 (1, 4096, 192) 的查询特征向量。此实现中未使用，因为InfoNCELoss会从k中自行选择。
        k: Tensor, 形状为 (1, 4096, 192) 的特征向量集合。
        image_name: str, 图像名称，用于加载对应的 positive 和 negative 索引文件。

    返回:
        loss: InfoNCE损失值 (标量 Tensor)。
    """
    # 获取不带后缀的文件名
    base_name = os.path.splitext(image_name)[0]

    # 修正路径并添加.npz后缀
    positive_index_path = os.path.join('/data/hyy/data_synapse/npy/info_label/positive_index/', base_name + '.npz')
    negative_index_path = os.path.join('/data/hyy/data_synapse/npy/info_label/negative_index/', base_name + '.npz')

    if not os.path.exists(positive_index_path) or not os.path.exists(negative_index_path):
        raise FileNotFoundError(f"Positive or negative index file not found for {base_name}.npz")

    # 从 npz 文件中加载索引
    positive_data = np.load(positive_index_path)
    negative_data = np.load(negative_index_path, allow_pickle=True)

    positive_indices = positive_data['positive_indices']
    negative_indices = negative_data['negative_indices']

    # 转换为 PyTorch 张量
    positive_indices_tensor = torch.from_numpy(positive_indices).long().to(k.device)
    # 负样本索引是对象数组，作为列表传递

    # 创建损失函数实例
    info_nce_loss_fn = InfoNCELoss(temperature=0.07)

    # 调用 InfoNCELoss 的 forward 方法
    loss = info_nce_loss_fn(q, k, positive_indices_tensor.unsqueeze(0), [negative_indices])

    return loss

# # 使用示例
# if __name__ == "__main__":
#     print("\n--- InfoNCELossPatch Example ---")
#
#     num_patches_patch_fn = 4096 # 假设 patch 数量为 4096
#     feat_dim_patch_fn = 192
#     q_dummy = torch.randn(1, num_patches_patch_fn, feat_dim_patch_fn)
#     k_dummy = torch.randn(1, num_patches_patch_fn, feat_dim_patch_fn)
#     image_name_dummy = 'CT_Abd_FLARE22_Tr_0040-084.npy' # 函数内部会处理后缀
#
#     print(f"Attempting to calculate loss for image: {image_name_dummy}")
#
#     try:
#         # 调用函数
#         patch_loss = InfoNCELossPatch(q_dummy, k_dummy, image_name_dummy)
#         print(f"InfoNCELossPatch calculated loss: {patch_loss.item():.4f}")
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         print("Please make sure you have run 'infoNCE_patch.py' and the index files exist at the specified path.")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")