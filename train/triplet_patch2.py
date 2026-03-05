import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# patch_vectors存储了4096个256维的向量，若每个256维的向量中的元素都为0则称为零向量。
# 若零向量在4096中占比超过95%则对零向量的索引进行随机采样1000个，对剩余向量的索引重复采样1000个。
# 若零向量在4096中占比小于等于95%则对零向量的索引进行随机采样2000个，对剩余向量的索引重复采样2000个。

# 定义路径
input_dir = '../../data/npy_medsam/CT_Abd/gts'
anchor_output_dir = '../../data/npy_medsam/CT_Abd/label_gts2/anchor_index'
positive_output_dir = '../../data/npy_medsam/CT_Abd/label_gts2/positive_index'
negative_output_dir = '../../data/npy_medsam/CT_Abd/label_gts2/negative_index'

# 创建输出目录
os.makedirs(anchor_output_dir, exist_ok=True)
os.makedirs(positive_output_dir, exist_ok=True)
os.makedirs(negative_output_dir, exist_ok=True)

# 设置阈值
threshold = 0.7

# 遍历 gts 文件夹中的所有 .npy 文件
for filename in os.listdir(input_dir):
    if filename.endswith('.npy'):
        # 加载标签矩阵
        file_path = os.path.join(input_dir, filename)
        label_matrix = np.load(file_path)
        assert label_matrix.shape == (1024, 1024), f"Input label matrix {filename} should have shape (1024, 1024)"

        # 分割成 16 x 16 的 patch 块
        patch_size = 16
        num_patches_per_dim = label_matrix.shape[0] // patch_size
        patches = []
        for i in range(num_patches_per_dim):
            for j in range(num_patches_per_dim):
                patch = label_matrix[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                patches.append(patch.flatten())

        # 转为数组计算余弦相似度
        patch_vectors = np.array(patches)
        num_patches = patch_vectors.shape[0]

        # 识别零向量和非零向量
        is_zero_vector = np.all(patch_vectors == 0, axis=1)
        zero_indices = np.where(is_zero_vector)[0]
        non_zero_indices = np.where(~is_zero_vector)[0]
        print(f"Identified {len(zero_indices)} zero vectors and {len(non_zero_indices)} non-zero vectors")

        # 锚点样本索引采样规则
        if len(zero_indices) / patch_vectors.shape[0] > 0.95:
            sampled_zero_indices = np.random.choice(zero_indices, size=1000, replace=False)
            sampled_non_zero_indices = np.random.choice(non_zero_indices, size=1000, replace=True)
            #print("More than 95% of vectors are zero vectors. Sampled 1000 zero and 1000 non-zero vectors.")
        else:
            sampled_zero_indices = np.random.choice(zero_indices, size=2000, replace=False)
            sampled_non_zero_indices = np.random.choice(non_zero_indices, size=2000, replace=True)
            #print("Less than or equal to 95% of vectors are zero vectors. Sampled 2000 zero and 2000 non-zero vectors.")

        anchor_indices = np.concatenate([sampled_zero_indices, sampled_non_zero_indices])
        print(f"Anchor indices count: {len(anchor_indices)}")

        # 保存锚点索引
        anchor_output_path = os.path.join(anchor_output_dir, filename)
        np.save(anchor_output_path, anchor_indices)
        #print(f"Saved anchor indices to {anchor_output_path}")


        # ###a.定义相似度：相同位置上标签相同所占百分比
        # similarity_matrix = np.zeros((num_patches, num_patches), dtype=np.float32)
        # for i in range(num_patches):
        #     similarity_matrix[i, :] = np.sum(patch_vectors == patch_vectors[i], axis=1) / patch_vectors.shape[1]

        ###b.余弦相似度
        unique_labels = np.unique(label_matrix)
        num_classes = max(unique_labels) + 1
        # 创建独热编码数组 (4096, 256, num_classes)
        one_hot_vectors = np.eye(num_classes)[patch_vectors]  # 每个元素编码为独热向量
        # 重塑为 (4096, 256 * num_classes)
        one_hot_patch_vectors = one_hot_vectors.reshape(patch_vectors.shape[0], -1)
        similarity_matrix = cosine_similarity(one_hot_patch_vectors)


        # 初始化 positive 和 negative 列表
        positive = []
        negative = []

        # 遍历每一行
        for idx in anchor_indices:
            row = similarity_matrix[idx]
            # 找到大于阈值的列索引
            positive_candidates = np.where((row > threshold) & (np.arange(len(row)) != idx))[0]
            # 找到小于或等于阈值的列索引
            negative_candidates = np.where((row <= threshold) & (np.arange(len(row)) != idx))[0]

            # 随机选择符合条件的索引
            if positive_candidates.size > 0:
                positive.append(np.random.choice(positive_candidates))
            else:
                positive.append(idx)  # 如果没有符合条件的值，存储 -1

            if negative_candidates.size > 0:
                negative.append(np.random.choice(negative_candidates))
            else:
                negative.append(idx)  # 如果没有符合条件的值，存储 -1

        # 将 positive 和 negative 转为数组
        positive = np.array(positive)
        negative = np.array(negative)

        # 保存结果
        positive_output_path = os.path.join(positive_output_dir, filename)
        negative_output_path = os.path.join(negative_output_dir, filename)

        np.save(positive_output_path, positive)
        np.save(negative_output_path, negative)

        positive_count = Counter(positive)
        negative_count = Counter(negative)
        print('没有正样本的patch数：', positive_count[-1])
        print('没有负样本的patch数：', negative_count[-1])

