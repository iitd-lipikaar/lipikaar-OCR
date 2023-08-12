# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import transformer

from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Mlp
import torchvision
import torchvision.models as models
from timm.models.mobilenetv3 import MobileNetV3, mobilenetv3_large_100

#from transformers import EfficientFormerConfig, EfficientFormerModel

from timm.models.efficientformer import EfficientFormer, EfficientFormerStage
#from timm.models.efficientformer
#from timm.models.efficientformer import PatchEmbed as ptEmbed

from .transformer import (PositionalEncoding,
                                 TransformerEncoder,
                                 TransformerEncoderLayer)

#'efficientformer_l1', 'efficientformer_l3', 'efficientformer_l7'


class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
       This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward_stream(self, tgt: Tensor, tgt_norm: Tensor, tgt_kv: Tensor, memory: Tensor, tgt_mask: Optional[Tensor],
                       tgt_key_padding_mask: Optional[Tensor]):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        tgt2, sa_weights = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask,
                                          key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None, update_content: bool = True):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)[0]
        if update_content:
            content = self.forward_stream(content, content_norm, content_norm, memory, content_mask,
                                          content_key_padding_mask)[0]
        return query, content

# class ConvBnRelu(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
#         super(ConvBnRelu, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

# class MobiNet(nn.Module):

#     def __init__(self, input_channels=3):
#         super(MobiNet, self).__init__()
#         self.input_channels = input_channels
        
#         # define MobileNetV2 with pretrained weights
#         # mobilenet = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
#         mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
#         self.features = mobilenet.features

#         self.no_weight_decay = None

#         # define additional layers
#         self.convbnrelu = ConvBnRelu(in_channels=576, out_channels=576, kernel_size=(3,3), stride=(2,1))
#         self.pos_encoder = PositionalEncoding(576, max_len=8*16)
#         encoder_layer = TransformerEncoderLayer(d_model=576, nhead=8,
#                 dim_feedforward=2048, dropout=0.1, activation='relu')
#         self.transformer = TransformerEncoder(encoder_layer, num_layers=3)
        
#     def forward(self, x):
#         x = self.features(x)
#         feature = self.convbnrelu(x)
        
#         n, c, h, w = feature.shape
#         feature = feature.view(n, c, -1).permute(2, 0, 1)
#         feature = self.pos_encoder(feature)
#         feature = self.transformer(feature)
#         feature = feature.permute(1, 2, 0).view(n, c, h, w)
#         feature = rearrange(feature, 'b c h w -> b (w h) c')
        
#         return feature

class ABI_MobileNet(nn.Module):

    def __init__(self):
        super(ABI_MobileNet, self).__init__()
        
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet_features = nn.Sequential(*list(self.mobilenet.features.children())[:-1])
        self.convbnrelu = ConvBnRelu(in_channels=320, out_channels=320, kernel_size=(3,3), stride=(2,1))

        self.pos_encoder = PositionalEncoding(320, max_len=8*16)
        encoder_layer = TransformerEncoderLayer(d_model=320, nhead=8,
                dim_feedforward=2048, dropout=0.1, activation='relu')
        self.transformer = TransformerEncoder(encoder_layer, num_layers=3)
        self.no_weight_decay = ['bn1', 'layer1']

    def forward(self, x):
        x = self.mobilenet_features(x)
        feature = self.convbnrelu(x)
        
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).permute(2, 0, 1)
        feature = self.pos_encoder(feature)
        feature = self.transformer(feature)
        feature = feature.permute(1, 2, 0).view(n, c, h, w)
        feature = rearrange(feature, 'b c h w -> b (w h) c')
        
        return feature

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv=nn.Conv2d, stride=2, inplace=True):
        super().__init__()
        p_size = [int(k//2) for k in kernel_size]
        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(query, content, memory, query_mask, content_mask, content_key_padding_mask,
                                 update_content=not last)
        query = self.norm(query)
        return query

def MobileNet():
    return ABI_MobileNet()

class EncoderMob(MobileNetV3):

    def __init__(self, img_size=224, in_chans=3, num_classes=0):
        super().__init__(features_only=True, num_classes=num_classes, in_chans=in_chans,
                         dropout_rate=0., drop_connect_rate=0., act_layer='nn.ReLU',
                         norm_layer='nn.BatchNorm2d', se_layer=True,
                         global_pool='avg', block_cfg=(4, 8, 16, 16, 8))

    def forward(self, x):
        # Return all tokens
        return self.forward_features(x)


# class EncoderEffFormer(nn.Module):
    
#     def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
#                  qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
#         super().__init__()
        
#         self.embed_layer = embed_layer(in_chans=3, embed_dim=embed_dim, img_size=img_size, patch_size=patch_size)
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_layer.num_patches + 1, embed_dim))
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.dropout = nn.Dropout(drop_rate)
#         self.transformer = nn.Transformer(
#             d_model=embed_dim, nhead=num_heads, num_encoder_layers=depth, dim_feedforward=int(embed_dim*mlp_ratio),
#             dropout=drop_rate, activation='gelu'
#         )
#         self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
#     def no_weight_decay(self):
#         return [name for name, param in self.named_parameters() if 'layernorm' in name or 'bias' in name]

#     def forward(self, x):
#         x = self.embed_layer(x)
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_token, x), dim=1)
#         # x = x + self.pos_embed
#         x = x + self.pos_embed[:, :x.shape[1], :]
#         x = self.dropout(x)
#         tgt = torch.zeros_like(x)
#         x = self.transformer(x, tgt)
#         x = x.mean(dim=1)
#         x = self.norm(x)
#         print('x shape in eff former: ', x.shape)
#         return x

EfficientFormer_width = {
    'l1': (48, 96, 224, 448),
    'l3': (64, 128, 320, 512),
    'l7': (96, 192, 384, 768),
    'l_sub': (384, 384, 384, 384),
    'new': (32, 64, 192, 384),
}

EfficientFormer_depth = {
    'l1': (3, 2, 6, 4),
    'l3': (4, 4, 12, 6),
    'l7': (6, 6, 18, 8),
    'l_sub': (2, 2, 6, 4),
    'new' : (2, 2, 2, 2),
}

class EncoderEffFormer(EfficientFormer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed):
        # super().__init__(img_size=img_size, in_chans=in_chans, num_classes=0,
        #                  depths=EfficientFormer_width['l7'], mlp_ratio=mlp_ratio, embed_dims = EfficientFormer_depth['l7'],
        #                  drop_rate=drop_rate, drop_path_rate=drop_path_rate)

        super().__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dims=EfficientFormer_width['l_sub'], depths=EfficientFormer_depth['new'],
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, embed_layer=embed_layer, downsample=False#, global_pool='token'
        )
        
        # self.backbone = EfficientFormer(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dims=EfficientFormer_depth['l_sub'], depths=EfficientFormer_width['l_sub'],
        #     num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
        #     attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, embed_layer=embed_layer#, global_pool='token'
        # )

        # with global_pool='token' : torch.Size([512, 4, 1, 1000])
        # without global_pool : torch.Size([512, 1, 1000])

        # 1D convolutional layers
        self.conv1 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=(1,2), padding=1)
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # self.global_pool = nn.AdaptiveAvgPool1d(384)
        #self.fc = nn.Linear(embed_dim, 384)
    
    def no_weight_decay(self):
        return [name for name, param in self.named_parameters() if 'layernorm' in name or 'bias' in name]

    def forward(self, x):
        # Return all tokens
        #print('Before passing to eff former: ', x.shape)
        y = self.forward_features(x)
        #print('Forward of effFormer: ', y.shape)
        y = self.conv1(y)
        #print('Forward of effFormer after conv: ', y.shape)
        y = y.reshape((y.shape[0], y.shape[1], y.shape[2]*y.shape[3]))
        #print('final features- ', y.shape)
        y = y.permute(0,2,1)
        #print(y.shape)
        return y

    def forward2(self, x):
        # Apply the EfficientFormer backbone
        x = self.backbone(x)
        #print('After applying backbone: ', x.shape)
        # num_classes = 1000, hidden_size = 128x384. 

        x = self.conv1(x)
        #print('After applying conv: ', x.shape)
        # x = self.conv2(x)

        # # Apply global average pooling
        # x = self.global_pool(x)  # Apply global average pooling and remove the last dimension

        #print('After applying global_pool: ', x.shape)

        return x



# class Encoder(EfficientFormerModel):

#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
#                  qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed):
#         super().__init__(depths=[depth] hidden_sizes: typing.List[int] = [48, 96, 224, 448]
#         downsamples: typing.List[bool] = [True, True, True, True]
#         dim: int = 448 key_dim: int = 32 attention_ratio: int = 4 resolution: int = 7 num_hidden_layers: int = 5 
#         num_attention_heads: int = 8 mlp_expansion_ratio: int = 4 hidden_dropout_prob: float = 0.0 patch_size: int = 16
#         num_channels: int = 3 pool_size: int = 3 downsample_patch_size: int = 3 downsample_stride: int = 2 downsample_pad: int = 1
#         drop_path_rate: float = 0.0 num_meta3d_blocks: int = 1 distillation: bool = Trueuse_layer_scale: bool = True
#         layer_scale_init_value: float = 1e-05 hidden_act: str = 'gelu' initializer_range: float = 0.02
#         layer_norm_eps: float = 1e-12**kwargs ))

#     def forward(self, x):
#         # Return all tokens
#         # print('Encoder: ',x.shape)
#         # Encoder:  torch.Size([512, 3, 32, 128])
#         y = self.forward_features(x)
#         # y.shape = torch.Size([512, 128, 384])
#         print(y.shape)
#         return y

class Encoder(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed):
        super().__init__(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         num_classes=0, global_pool='', class_token=False)  # these disable the classifier head

    def forward(self, x):
        # Return all tokens
        # print('Encoder: ',x.shape)
        # Encoder:  torch.Size([512, 3, 32, 128])
        y = self.forward_features(x)
        # y.shape = torch.Size([512, 128, 384])
        return y


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)
