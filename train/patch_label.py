import numpy as np
import cv2

img_path = '../../data/npy_litemedsam/train/gts/CT_Abd_FLARE22_Tr_0001-049.npy'  # (512, 512)
img_path = '../../data/npy_medsam/CT_Abd/gts/CT_Abd_FLARE22_Tr_0001-049.npy'  # (1024, 1024)
img = np.load(img_path)
# 下采样
target_shape = (64, 64)  # 注意 OpenCV 需要 (宽, 高) 顺序
downsampled_img = cv2.resize(img, target_shape, interpolation=cv2.INTER_NEAREST)
print(img.max(), img.min(), np.unique(img), img.shape)
print(np.unique(downsampled_img), downsampled_img.shape)

# 1. 将下采样的标签矩阵展开成 4096 维向量
vectorized_label = downsampled_img.flatten()  # (4096,)

# 2. 构建 (4096, 4096) 矩阵
# 对每个元素进行比较，标签值相同记为 1，不同记为 -1
# comparison_matrix = np.ones((vectorized_label.size, vectorized_label.size), dtype=int)
# for i in range(vectorized_label.size):
#     for j in range(vectorized_label.size):
#         comparison_matrix[i, j] = 1 if vectorized_label[i] == vectorized_label[j] else -1

# 2. 构建 (4096, 4096) 矩阵，使用矢量化实现
# np.equal.outer 生成布尔矩阵，比较是否相等
comparison_matrix = np.equal.outer(vectorized_label, vectorized_label)  # 比较是否相等
# np.where 转换布尔值为 1 和 -1
comparison_matrix = np.where(comparison_matrix, 1, -1)

# 查看结果
print(vectorized_label, np.unique(vectorized_label))  # (4096,)
print(comparison_matrix, np.unique(comparison_matrix))  # (4096, 4096)