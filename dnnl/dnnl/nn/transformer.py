import copy
from collections.abc import Callable

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import MultiheadAttention

__all__ = [
    'Transformer',
    'TransformerEncoder',
    'TransformerEncoderLayer',
    'TransformerDecoder',
    'TransformerDecoderLayer',
]


def _get_activation_fn(
    activation: str | Callable[[Tensor], Tensor],
) -> Callable[[Tensor], Tensor]:
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if callable(activation):
        return activation
    raise ValueError('The activation must be `relu`, `gelu`, or a callable.')


def _clone_module(module: nn.Module, num_layers: int) -> nn.ModuleList:
    return nn.ModuleList(copy.deepcopy(module) for _ in range(num_layers))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            bias=bias,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        if self.norm_first:  # Pre-LN
            x = src + self._sa_block(
                self.norm1(src),
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
            x = x + self._ff_block(self.norm2(x))
            return x
        else:  # Post-LN
            x = self.norm1(
                src + self._sa_block(
                    src,
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                )
            )  # fmt: skip
            x = self.norm2(x + self._ff_block(x))
            return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool,
    ) -> Tensor:
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )  # fmt: skip
        x = self.dropout1(x)
        return x

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ):
        super().__init__()
        self.layers = _clone_module(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        bias: bool = True,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            bias=bias,
        )
        self.mha_attn = MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            bias=bias,
        )
        self.multihead_attn = self.mha_attn

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        if self.norm_first:
            x = tgt + self._sa_block(
                self.norm1(tgt),
                tgt_mask,
                tgt_key_padding_mask,
                tgt_is_causal,
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            return x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                tgt + self._sa_block(
                    tgt,
                    tgt_mask,
                    tgt_key_padding_mask,
                    tgt_is_causal,
                )
            )  # fmt: skip
            x = self.norm2(
                x + self._mha_block(
                    x,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    memory_is_causal,
                )
            )  # fmt: skip
            return self.norm3(x + self._ff_block(x))

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool,
    ) -> Tensor:
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )  # fmt: skip
        return self.dropout1(x)

    def _mha_block(
        self,
        x: Tensor,
        memory: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool,
    ) -> Tensor:
        x = self.mha_attn(
            x,
            memory,
            memory,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        return self.dropout2(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout3(x)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ):
        super().__init__()
        self.layers = _clone_module(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        encoder_layer = TransformerEncoderLayer(
            d_model,
            num_heads,
            dim_feedforward,
            bias=bias,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
        )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            norm=encoder_norm,
        )

        decoder_layer = TransformerDecoderLayer(
            d_model,
            num_heads,
            dim_feedforward,
            bias=bias,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
        )
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            norm=decoder_norm,
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        src_is_causal: bool = False,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise AssertionError(
                'the feature number of src and tgt must be equal to d_model'
            )
        if src.size(0) != tgt.size(0):
            raise AssertionError('the batch size of src and tgt must be equal')

        memory = self.encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=src_is_causal,
        )
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )
        return output
