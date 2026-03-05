import torch
# 将蒸馏得到的image_encoder与prompt_encoder和decoder进入融合

# 加载预训练模型权重
lite_encoder_state_dict = torch.load('../../synapse_result/distill/mse+mse_0.1triplet_0.01tr_0.3_100epoch/')
lite_encoder_state_dict = lite_encoder_state_dict['model']

medsam_vit_b_state_dict = torch.load('sam_vit_b.pth')

new_state_dict = {}

for key in lite_encoder_state_dict.keys():
    new_key = 'image_encoder.' + key
    new_state_dict[new_key] = lite_encoder_state_dict[key]

for key, value in medsam_vit_b_state_dict.items():
    if key.startswith('prompt_encoder') or key.startswith('mask_decoder'):
        new_state_dict[key] = value

torch.save(new_state_dict, '../../synapse_result/distill/mse+mse_0.1triplet_0.01tr_0.3_100epoch/')

