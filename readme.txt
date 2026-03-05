一、训练（蒸馏）
1、distill_TransSC：蒸馏代码【运行1】【1处要改】
2、checkpoint_model：将蒸馏的image_encoder与sam的prompt_encoder和decoder融合。

TransSC_modeling：模型代码
TransSC_configs：模型参数

利用蒸馏的三段代码得到lite_TransSC_medsam.pth

二、推理
1、TransSC_infer.py：结果保存在test_demo_self/XXX/segs中（其中XXX为分割的结果类型有四种：2个train和2个蒸馏的结果）【所有的结果都在test_demo_self中】【3处要改】
TransSC_modeling.py：模型结构
TransSC_configs.py：模型参数

gts中为掩码标签，含有键gts
imgs中有原图和box，含有键imgs和boxes
segs中为掩码结果，含有键segs

三、评估
1、evaluation/compute_metrics.py：得到dsc和nsd两个评价指标。保存为.csv文件，并保存在对应的test_demo_self的分割结果中（如../test_demo_self/distillation_400epoch/metrics_distillation_400epoch.csv）


五、triplet
triplet patch：用于存储negative和positive序号的代码（有两种方法）
triplet loss：蒸馏时会调用这个函数（改路径和最后的loss可以选择两种不同的方法）
