## 📌 项目结构说明

---

### 一、训练（蒸馏）

#### 🚀 训练代码
| 文件 | 说明 |
|------|------|
| `triplet_distill_vit.py` | 训练（蒸馏）代码，学生模型为 ViT，损失函数：三元组损失 |
| `triplet_distill_tinyvit.py` | 训练（蒸馏）代码，学生模型为 TinyViT，损失函数：三元组损失 |
| `info_distill_TransSC.py` | 训练（蒸馏）代码，损失函数：InfoNCE 损失 |
| `checkpoint_model.py` | 将蒸馏后的 image_encoder 与 SAM 的 prompt_encoder、decoder 融合 |

#### 🔧 三元组损失相关
| 文件 | 说明 |
|------|------|
| `triplet_patch.py` | 用于存储 negative 和 positive 序号的代码 |
| `triplet_loss.py` | 蒸馏时调用的三元组损失函数 |

#### 🔧 InfoNCE损失相关
| 文件 | 说明 |
|------|------|
| `infoNCE_patch.py` | 用于存储 negative 和 positive 序号的代码 |
| `info_nce_loss.py` | 蒸馏时调用的infoNCE损失函数 |

#### 🧠 模型定义
| 文件 | 说明 |
|------|------|
| `TransSC_modeling.py` | vit模型结构定义 |
| `TransSC_configs.py` | vit模型参数配置 |

| 文件| 说明 |
|------|------|
| `segment_anything` | sam模型结构代码文件夹 |
| `tinyvit/tiny_vit_sam.py` | tinyvit模型代码 |

---

### 二、推理

#### 📁 文件说明
| 文件/文件夹 | 说明 |
|------------|------|
| `TransSC_infer.py` | 推理代码 |
| `TransSC_modeling.py` | 模型结构 |
| `TransSC_configs.py` | 模型参数 |
| `tiny_vit_sam.py` | tinyvit模型代码 |

### 三、评估
1、evaluation/compute_metrics.py：得到dsc和nsd两个评价指标。保存为.csv文件。

