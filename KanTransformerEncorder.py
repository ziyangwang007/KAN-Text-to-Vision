#!/usr/bin/env python
# -*-coding:utf-8 -*-
import copy

import torch.nn.functional as F
from typing import Union, Callable, List, Optional

from torch import Tensor, nn
from torch.nn.modules.transformer import TransformerEncoderLayer
from torch.nn import TransformerEncoder
from fasterkan import FasterKAN as KAN


class KANTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None, hdim_kan=192):
        super(KANTransformerEncoderLayer, self).__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps,
            batch_first=batch_first, norm_first=norm_first, dtype=dtype, device=device,
        )
        # Replace the feed forward network with KAN
        self.linear1 = None  # Removing unnecessary linear layer
        self.linear2 = None  # Removing unnecessary linear layer
        hidden_dim = hdim_kan if hdim_kan and hdim_kan > 0 else max(1, d_model // 2)
        self.kan = KAN([d_model, hidden_dim, d_model])

    def _ff_block(self, x):
        b, t, d = x.shape
        # Use KAN as the feed-forward network
        return self.dropout(self.kan(x.reshape(-1, x.shape[-1])).reshape(b, t, d))


class MixLayerTransformerEncoder(TransformerEncoder):
    def __init__(self, layers: Optional[List[TransformerEncoderLayer]]=None, **kwargs):
        if layers is not None:
            # reusing the same parameters as TransformerEncoder
            norm = kwargs.pop('norm', None)
            enable_nested_tensor = kwargs.pop('enable_nested_tensor', True)
            mask_check = kwargs.pop('mask_check', True)
            dummy_layer = layers[0] if len(layers) > 0 else nn.TransformerEncoderLayer(d_model=2, nhead=1)
            super(MixLayerTransformerEncoder, self).__init__(encoder_layer=dummy_layer, num_layers=len(layers),
                                                             **kwargs)
            # redefine the layers
            self.layers = nn.ModuleList([copy.deepcopy(layer) for layer in layers])
            self.num_layers = len(layers)
            self.norm = norm
            self.enable_nested_tensor = enable_nested_tensor
            self.use_nested_tensor = False
            self.mask_check = mask_check
        else:
            # if layers is None, use the default TransformerEncoder
            super().__init__(**kwargs)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
