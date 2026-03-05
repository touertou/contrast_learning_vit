# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.functional as F

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
#from scipy import ndimage
import triplet_TransSC_configs as configs
# from .vit_seg_modeling_resnet_skip import ResNetV2

device = ('cuda:1' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

logger = logging.getLogger(__name__)

# 定义了模型中一些用于加载预训练权重时的字符串常量，代表不同层的权重名称
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


# 将 NumPy 格式的权重转换为 PyTorch Tensor
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


# 定义 Swish 激活函数
def swish(x):
    return x * torch.sigmoid(x)


# 定义一个激活函数字典
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = _pair(img_size)
        self.num_attention_heads = config.transformer["num_heads"]

        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=192,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 192))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):  # (n,3,224,224)
        x = self.patch_embeddings(x)  # (n,768,14,14)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (n, 196, 768)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Pretreatment(nn.Module):
    def __init__(self, config):
        super(Pretreatment, self).__init__()
        self.pretreatment = Linear(in_features=192, out_features=256)

    def forward(self, x):
        x = self.pretreatment(x)
        return x


class Attention(nn.Module):
    def __init__(self, config, vis, is_test, is_last):
        super(Attention, self).__init__()
        self.is_last = is_last
        self.vis = vis
        self.is_test = is_test
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 768
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        #self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):  # (n, 196, 768)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (n, 196, 1, 768)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (n, 6, 196, 128)

    def forward(self, hidden_states):  # (n, 196, 384)
        mixed_query_layer = self.query(hidden_states)  # (n, 4096, 192)
        #mixed_key_layer = self.key(hidden_states)  # (n, 4096, 192)
        mixed_key_layer = mixed_query_layer
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (n, 1, 196, 768)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.is_test:
            tr = 0
            Regular_term = 0
        else:
            attention_scores1 = torch.matmul(query_layer, key_layer.transpose(-1,
                                                                              -2))  # (n, 1, 196, 768) *  (n, 1, 768, 196) = (n, 1, 196, 196)
            attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
            attention_probs1 = self.softmax(attention_scores1)  # patches对于其他patches的 attention 系数
            attention_probs = (attention_probs1 + attention_probs1.transpose(-1, -2)) / 2  # (1,12,4096,4096)

            I_head_size = torch.eye(self.attention_head_size).to(device)
            V = F.normalize(value_layer, p=2, dim=2)

            adjacency_matrix_vector = attention_probs.sum(axis=3)
            D = torch.diag_embed(adjacency_matrix_vector)
            Laplace_matrix = D - attention_probs

            # # 1.最小化迹逼近矩阵L的前k个特征向量
            # target_matrix = Laplace_matrix

            # 2.最小化迹逼近矩阵L_sym的前k个特征向量
            #             adjacency_matrix_vector_1 = adjacency_matrix_vector**(-1/2)
            #             D_sq = torch.diag_embed(adjacency_matrix_vector_1)
            #             L_sym_1 = torch.matmul(Laplace_matrix, D_sq)
            #             L_sym = torch.matmul(D_sq, L_sym_1)
            #             target_matrix = L_sym

            # 最小化迹逼近矩阵L_rw的前k个特征向量
            adjacency_matrix_vector_1 = adjacency_matrix_vector ** (-1)
            D_1 = torch.diag_embed(adjacency_matrix_vector_1)
            L_rw = torch.matmul(D_1, Laplace_matrix)
            target_matrix = L_rw

            V_T = V.permute(0, 1, 3, 2)
            VV_T = (torch.matmul(V_T, V) - I_head_size) ** 2
            Regular_term = VV_T.sum()
            target_matrix_V = torch.matmul(target_matrix, V)
            VT_target_matrix_V = torch.matmul(V_T, target_matrix_V)
            VT_target_matrix_V_TR = torch.sum(VT_target_matrix_V, [0, 1])
            tr = abs(VT_target_matrix_V_TR.trace())
            tr = tr / (self.attention_head_size * self.num_attention_heads * hidden_states.shape[0])
            Regular_term = Regular_term / (self.attention_head_size * self.num_attention_heads * hidden_states.shape[0])

            # attention_probs = self.attn_dropout(attention_probs)
        if self.is_last:
            context_layer = value_layer
        else:
            context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (12,196,12,32)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, tr, Regular_term, mixed_query_layer, mixed_key_layer
        #return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, 4 * config.hidden_size)
        self.fc2 = Linear(4 * config.hidden_size, config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, vis, is_test, is_last):
        super(Block, self).__init__()
        self.is_last = is_last
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis, is_test, is_last)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, tr, Regular_term, mixed_query_layer, mixed_key_layer = self.attn(x)
        if self.is_last:
            x = x
        else:
            x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, tr, Regular_term, mixed_query_layer, mixed_key_layer

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis, is_test, is_last):
        super(Encoder, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.is_last = is_last
        self.is_test = is_test
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer["num_layers"]):
            if i == config.transformer["num_layers"] - 1:
                layer = Block(config, vis, is_test, is_last=True)
            else:
                layer = Block(config, vis, is_test, is_last=False)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states, tr, Regular_term, mixed_query_layer, mixed_key_layer = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return hidden_states, tr, Regular_term, mixed_query_layer, mixed_key_layer


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, is_test, is_last):
        super(Transformer, self).__init__()
        self.config = config
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, is_test, is_last)

        #self.Pretreatment = Pretreatment(config)
        self.neck = nn.Sequential(
            nn.Conv2d(
                192,
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

    def forward(self, input_ids):
        h, w = input_ids.shape[2], input_ids.shape[3]
        patch_size = _pair(self.config.patches["size"])
        embedding_output = self.embeddings(input_ids)  # (1,3,256,256) -> (1,64,192) # (256/32)*(256/32)=8*8=64
        #embedding_output = self.Pretreatment(embedding_output)
        x, tr, Regular_term, mixed_query_layer, mixed_key_layer = self.encoder(embedding_output)  # x:(1,64,192)
        n_patches = int(h / patch_size[0])
        x = x.reshape(x.shape[0], n_patches, n_patches, x.shape[2])
        x = self.neck(x.permute(0, 3, 1, 2))
        return x, tr, Regular_term, mixed_query_layer, mixed_key_layer


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, zero_head=False, vis=False, is_test=False,
                 is_last=False):
        super(VisionTransformer, self).__init__()

        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis, is_test, is_last)
        # self.decoder = DecoderCup(config)
        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, tr, Regular_term, mixed_query_layer, mixed_key_layer = self.transformer(x)
        # x = self.decoder(x, features=None)
        # logits = self.segmentation_head(x)
        # return logits, tr, Regular_term
        # B, _, C = x.size()
        # x = x.view(B, 64, 64, C)  # (b,64,64,320)
        # x = x.permute(0, 3, 1, 2)  # (b,320,64,64)

        # return x, tr, Regular_term, mixed_query_layer, mixed_key_layer  # train
        return x  # test

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


