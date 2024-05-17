__all__ = ['PatchTST']

from torch import nn
from torch import Tensor
from typing import Callable, Optional

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp

class Model(nn.Module):

    def __init__(
        self,
        decomposition,
        encoder_input_size,  # channels: 1 in our case
        # patching
        patch_len: int,  # P
        stride: int,  # S
        context_window: int,  # L
        target_window: int,  # T
        padding_patch=None,
        # normalization
        use_RevIN=True,  # use RevIN to normalize
        affine=True,  # learnable affine parameters?
        subtract_last=False,  #
        # Projection
        dropout: float = 0.0,
        pe: str = "zeros",
        learn_pe: bool = True,
        # transformer
        n_layers: int = 3,
        d_model=128,
        n_heads=16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        pre_norm: bool = False,
        act: str = "gelu",
        res_attention: bool = True,
        store_attn: bool = False,
        # head
        individual=False,  # individual layers for each var?
        head_dropout=0,
    ):
        super().__init__()

        self.decomposition = decomposition

        if self.decomposition:
            # ! todo
            pass
        else:
            self.model = PatchTST_backbone(
                encoder_input_size=encoder_input_size,
                patch_len=patch_len,
                stride=stride,
                context_window=context_window,
                target_window=target_window,
                use_RevIN=use_RevIN,
                affine=affine,
                dropout=dropout,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                pre_norm=pre_norm,
                act=act,
                res_attention=res_attention,
                store_attn=store_attn,
                individual=individual,
                head_dropout=head_dropout,
            )

    def forward(self, x):  # x: [Batch, Input length, Channel]
        if self.decomposition:
            # ! todo
            pass
        else:
            x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x
