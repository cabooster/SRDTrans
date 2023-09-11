import torch
from torch import nn
from einops import rearrange
from SRDTrans.SpatioTemporalTrans.TemporalTrans import TemporalTransLayer, LearnedPositionalEncoding
from SRDTrans.SpatioTemporalTrans.SpatioiTrans import SpatioTransLayer


class TemporalTransformer(nn.Module):
    def __init__(
            self,
            seq_length,
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            input_dropout_rate,
            attn_dropout_rate,
    ):
        super(TemporalTransformer, self).__init__()
        self.transformer = TemporalTransLayer(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=hidden_dim,
            dropout_rate=input_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        self.position_encoding = LearnedPositionalEncoding(
            embedding_dim, seq_length
        )
        self.pre_dropout = nn.Dropout(p=input_dropout_rate)
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = rearrange(x, 'b c s h w -> (b h w) s c')
        x = self.position_encoding(x)
        x = self.pre_dropout(x)
        x = self.transformer(x)
        x = self.pre_head_ln(x)
        x = rearrange(x, '(b p1 p2) s c -> b c s p1 p2', p1=H, p2=W)
        return x


class SpatioTransformer(nn.Module):
    def __init__(
            self,
            embedding_dim,
            num_layers,
            num_heads,
            window_size,
            hidden_dim,
            attn_drop=0.,
            input_drop=0.,
    ):
        super(SpatioTransformer, self).__init__()
        self.transformer = SpatioTransLayer(
            dim=embedding_dim,
            depth=num_layers,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=hidden_dim/embedding_dim,
            qkv_bias=True,
            qk_scale=None,
            drop=input_drop,
            attn_drop=attn_drop,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
        )

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = rearrange(x, 'b c s h w -> (b s) (h w) c')
        x = self.transformer(x, H, W)
        x = rearrange(x, '(b p1) (h p2) c -> b c p1 h p2', p1=D, p2=W)
        return x


class SpatioTemporalTrans(nn.Module):
    def __init__(
            self,
            seq_length,
            embedding_dim,
            num_heads,
            hidden_dim,
            space_window_size,
            attn_dropout_rate,
            input_dropout_rate,
            num_time_trans_layer=2,
            num_space_trans_layer=2,
    ):
        super(SpatioTemporalTrans, self).__init__()
        self.timeTrans = TemporalTransformer(
            seq_length=seq_length,
            embedding_dim=embedding_dim,
            num_layers=num_time_trans_layer,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            input_dropout_rate=input_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        self.spaceTrans = SpatioTransformer(
            embedding_dim=embedding_dim,
            num_layers=num_space_trans_layer,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            window_size=space_window_size,
            attn_drop=attn_dropout_rate,
        )

    def forward(self, x):
        x = self.timeTrans(x)
        x = self.spaceTrans(x)
        return x
