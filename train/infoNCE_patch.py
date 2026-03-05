import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

input_dir = '/home/hyy/workspace/data/data_synapse/npy/train/gts/'
positive_output_dir = '/data/hyy/data_synapse/npy/info_label/positive_index'
negative_output_dir = '/data/hyy/data_synapse/npy/info_label/negative_index'

# 创建输出目录
os.makedirs(positive_output_dir, exist_ok=True)
os.makedirs(negative_output_dir, exist_ok=True)

# 设置阈值
threshold = 0.9

# 遍历 gts 文件夹中的所有 .npy 文件
for filename in os.listdir(input_dir):
    if filename.endswith('.npy'):
        # 加载标签矩阵
        file_path = os.path.join(input_dir, filename)
        label_matrix = np.load(file_path)

        # 分割成 16 x 16 的 patch 块
        patch_size = 8  # 1024*1024对应16，512*512对应8
        num_patches_per_dim = label_matrix.shape[0] // patch_size
        patches = []
        for i in range(num_patches_per_dim):
            for j in range(num_patches_per_dim):
                patch = label_matrix[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                patches.append(patch.flatten())

        # 转为数组计算余弦相似度
        patch_vectors = np.array(patches)
        num_patches = patch_vectors.shape[0]

        ##自定义相似度：相同位置上标签相同所占百分比
        similarity_matrix = np.zeros((num_patches, num_patches), dtype=np.float32)  # (4096,4096)
        for i in range(num_patches):
            similarity_matrix[i, :] = np.sum(patch_vectors == patch_vectors[i], axis=1) / patch_vectors.shape[1]

        ## 余弦相似度
        # similarity_matrix = cosine_similarity(patch_vectors)

        # 初始化 positive 和 negative 列表
        positive = []
        negative = []

        # 遍历每一行
        for row in similarity_matrix:
            # 找到大于阈值的列索引
            positive_candidates = np.where(row > threshold)[0]
            # 找到小于或等于阈值的列索引
            negative_candidates = np.where(row <= threshold)[0]

            # 随机选择符合条件的索引
            if positive_candidates.size > 0:
                positive.append(np.random.choice(positive_candidates))
            else:
                positive.append(-1)  # 如果没有符合条件的值，存储 -1

            if negative_candidates.size > 0:
                negative.append(negative_candidates)
            else:
                negative.append(-1)  # 如果没有符合条件的值，存储 -1

        # 将 positive 和 negative 转为数组
        positive = np.array(positive)
        negative = np.array(negative, dtype=object)  # 使用dtype=object以支持不同长度的数组

        # 保存结果
        base_name = os.path.splitext(filename)[0]
        print(base_name)
        positive_output_path = os.path.join(positive_output_dir, base_name + '.npz')
        negative_output_path = os.path.join(negative_output_dir, base_name + '.npz')

        # 使用npz格式保存
        np.savez(positive_output_path, positive_indices=positive)
        np.savez(negative_output_path, negative_indices=negative)

        print(positive)
        print(negative)

