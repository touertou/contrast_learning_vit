import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import glob
from tqdm import tqdm
import torch.nn.functional as F


def generate_and_save_indices(data_root, output_root):
    """
    为所有图像生成并保存不同尺寸的正负样本索引

    Args:
        data_root: 包含gts文件夹的根目录
        output_root: 保存索引的根目录
    """
    # 创建不同尺寸的输出目录
    sizes = [(256, 256), (128, 128), (64, 64)]
    for H, W in sizes:
        os.makedirs(os.path.join(output_root, f"{H}x{W}"), exist_ok=True)

    # 获取所有标签文件
    gt_path = os.path.join(data_root, 'gts')
    gt_files = sorted(glob.glob(os.path.join(gt_path, '**/*.npy'), recursive=True))

    print(f"开始处理 {len(gt_files)} 个文件...")

    for gt_file in tqdm(gt_files):
        image_name = os.path.basename(gt_file)

        # 为每个尺寸生成索引
        for H, W in sizes:
            positive_indices, negative_indices = Triplet_patch_tinyvit(H, W, image_name)

            # 保存索引
            save_path = os.path.join(output_root, f"{H}x{W}", f"{os.path.splitext(image_name)[0]}_indices.npz")
            np.savez(save_path,
                     positive_indices=positive_indices,
                     negative_indices=negative_indices)

    print("索引生成完成！")


def Triplet_patch_tinyvit(H, W, image_name):
    # 写死的目录路径
    input_dir = '../../data/data_synapse/npy/train/gts/'

    # 构造完整路径
    file_path = os.path.join(input_dir, image_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    # 1. 加载标签图像
    label_matrix = np.load(file_path)
    # print(f"原始标签形状: {label_matrix.shape}")  # e.g., (512, 512)

    # 2. 调整标签大小为 H × W，使用最近邻插值
    resized_label = cv2.resize(label_matrix, (W, H), interpolation=cv2.INTER_NEAREST)
    # print(f"调整后的标签形状: {resized_label.shape}")  # e.g., (64, 64)

    # 3. 展平标签
    flat_label = resized_label.flatten()  # shape: (H*W,)

    # 4. 构建相似度矩阵
    similarity_matrix = (flat_label[:, None] == flat_label[None, :]).astype(np.uint8)
    # print(f"相似度矩阵形状: {similarity_matrix.shape}")  # e.g., (4096, 4096)

    # 5. 构造 positive 和 negative 索引列表
    positive_indices, negative_indices = [], []

    for i, row in enumerate(similarity_matrix):
        positive_candidates = np.where((row == 1) & (np.arange(len(row)) != i))[0]
        negative_candidates = np.where(row == 0)[0]

        pos_idx = np.random.choice(positive_candidates) if positive_candidates.size > 0 else -1
        neg_idx = np.random.choice(negative_candidates) if negative_candidates.size > 0 else -1

        positive_indices.append(pos_idx)
        negative_indices.append(neg_idx)

    return np.array(positive_indices), np.array(negative_indices)


# image_name = 'CT_Abd_0001-008.npy'
# positive_indices, negative_indices = Triplet_patch_tinyvit(H=64, W=64, image_name=image_name)

if __name__ == "__main__":
    # 设置数据路径
    data_root = "../../data/data_synapse/npy/train"
    output_root = "../../data/data_synapse/triplet_indices"

    # 生成并保存索引
    generate_and_save_indices(data_root, output_root)