from timm.models.efficientnet_blocks import get_width_and_height_from_size, \
    round_filters, resolve_bn_args, resolve_act_layer, drop_connect
from timm.models.efficientnet_builder import EfficientNetBuilder
from timm.models.efficientnet_builder import decode_arch_def, efficientnet_init_weights
from timm.models.efficientnet_builder import EfficientNetBuilder
from timm.models.efficientnet_builder import decode_arch_def, efficientnet_init_weights
from timm.models.efficientnet_builder import EfficientNetBuilder
from timm.models.efficientnet_builder import decode_arch_def, efficientnet_init_weights
from timm.models.efficientnet_builder import EfficientNetBuilder
from timm.models.efficientnet_builder import decode_arch_def, efficientnet_init_weights
from timm.models.efficientnet_builder import EfficientNetBuilder
from timm.models.efficientnet_builder import decode_arch_def, efficientnet_init_weights
from timm.models.vision_transformer import Mlp, PatchEmbed
import torch.nn as nn

class EncoderEffFormer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed):
        super().__init__()

        # Initialize the EfficientFormer backbone
        self.backbone = EfficientFormer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, embed_layer=embed_layer
        )
        
        # Add a global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final linear layer
        self.fc = nn.Linear(embed_dim, 384)

    def forward(self, x):
        # Apply the EfficientFormer backbone
        x = self.backbone(x)
        
        # Apply global average pooling
        x = x.permute(0, 2, 1)  # Convert from (batch_size, seq_length, hidden_size) to (batch_size, hidden_size, seq_length)
        x = self.global_pool(x).squeeze(-1)  # Apply global average pooling and remove the last dimension
        
        # Apply the final linear layer
        x = self.fc(x)

        return x
