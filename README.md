## 📌 项目结构说明

---

### 一、训练（蒸馏）

#### 🚀 训练代码
| 文件 | 说明 |
|------|------|
| `triplet_distill_vit.py` | 蒸馏代码，学生模型为 ViT，损失函数：三元组损失 |
| `triplet_distill_tinyvit.py` | 蒸馏代码，学生模型为 TinyViT，损失函数：三元组损失 |
| `info_distill_TransSC.py` | 蒸馏代码，损失函数：InfoNCE 损失 |
| `checkpoint_model.py` | 将蒸馏后的 image_encoder 与 SAM 的 prompt_encoder、decoder 融合 |

#### 🔧 三元组损失相关
| 文件 | 说明 |
|------|------|
| `triplet_patch.py` | 用于存储 negative 和 positive 序号的代码 |
| `triplet_loss.py` | 蒸馏时调用的三元组损失函数 |

#### 🧠 模型定义
| 文件 | 说明 |
|------|------|
| `TransSC_modeling.py` | 模型结构定义 |
| `TransSC_configs.py` | 模型参数配置 |

---

### 二、推理

#### 📁 文件说明
| 文件/文件夹 | 说明 |
|------------|------|
| `TransSC_infer.py` | 推理代码，结果保存在 `test_demo_self/XXX/segs` 中 |
| `TransSC_modeling.py` | 模型结构 |
| `TransSC_configs.py` | 模型参数 |

#### 📂 文件夹结构说明
###三、评估
1、evaluation/compute_metrics.py：得到dsc和nsd两个评价指标。保存为.csv文件。




