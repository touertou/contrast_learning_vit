# contrast_learning_vit
###一、训练（蒸馏）：train
1、triplet_distill_vit：蒸馏代码，学生模型为vit，损失函数三元组损失
2、triplet_distill_tinyvit：训练（蒸馏）代码，学生模型为tinyvit，损失函数三元组损失
3、info_distill_TransSC：训练（蒸馏）代码，损失函数infonce损失
4、checkpoint_model：将蒸馏的image_encoder与sam的prompt_encoder和decoder融合。
####三元组损失：
1、triplet patch：用于存储negative和positive序号的代码
2、triplet loss：蒸馏时会调用这个函数
####模型：
1、TransSC_modeling：模型代码
2、TransSC_configs：模型参数

###二、推理
1、TransSC_infer.py：结果保存在test_demo_self/XXX/segs中
TransSC_modeling.py：模型结构
TransSC_configs.py：模型参数

gts中为掩码标签，含有键gts
imgs中有原图和box，含有键imgs和boxes
segs中为掩码结果，含有键segs

###三、评估
1、evaluation/compute_metrics.py：得到dsc和nsd两个评价指标。保存为.csv文件。




