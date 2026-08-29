import re
from collections.abc import Mapping

from torch import Tensor

_BLOCK_KEY = re.compile(r'transformer\.h\.(\d+)\.(.+)')

_TOP_LEVEL_KEYS = {
    'transformer.wte.weight': 'tok_embed.weight',
    'transformer.wpe.weight': 'pos_embed.weight',
    'transformer.ln_f.weight': 'final_norm.weight',
    'transformer.ln_f.bias': 'final_norm.bias',
    'lm_head.weight': 'lm_head.weight',
}

_BLOCK_KEYS = {
    'ln_1.weight': 'norm1.weight',
    'ln_1.bias': 'norm1.bias',
    'attn.c_proj.weight': 'attn.attn.out_proj.weight',
    'attn.c_proj.bias': 'attn.attn.out_proj.bias',
    'ln_2.weight': 'norm2.weight',
    'ln_2.bias': 'norm2.bias',
    'mlp.c_fc.weight': 'mlp.fc1.weight',
    'mlp.c_fc.bias': 'mlp.fc1.bias',
    'mlp.c_proj.weight': 'mlp.fc2.weight',
    'mlp.c_proj.bias': 'mlp.fc2.bias',
}

_TRANSPOSED_BLOCK_WEIGHTS = {
    'attn.c_proj.weight',
    'mlp.c_fc.weight',
    'mlp.c_proj.weight',
}

_IGNORED_BLOCK_KEYS = {'attn.bias', 'attn.masked_bias'}


def state_dict_converter(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Convert a Hugging Face GPT-2 state dict for `GPT2Model`.

    Hugging Face stores the query, key, and value projections in one `Conv1D`
    layer and stores all `Conv1D` weights as `(in_features, out_features)`.
    The local model uses separate Q/K/V `Linear` layers whose weights have the
    usual `(out_features, in_features)` layout, so those tensors must be split
    and transposed in addition to renaming their keys.

    Args:
        state_dict (Mapping[str, Tensor]): State dict produced by
            `GPT2LMHeadModel.state_dict()`.

    Returns:
        state_dict (dict[str, Tensor]): A state dict whose keys and tensor layouts
            match the local `GPT2Model`.

    Raises:
        ValueError: If a fused Q/K/V tensor has an invalid shape.
        KeyError: If the source contains an unsupported parameter key.
    """
    converted = {}

    for key, value in state_dict.items():
        if target_key := _TOP_LEVEL_KEYS.get(key):
            converted[target_key] = value
            continue

        match = _BLOCK_KEY.fullmatch(key)
        if match is None:
            raise KeyError(f'Unsupported GPT-2 state-dict key: {key!r}')

        layer_index, block_key = match.groups()
        target_prefix = f'backbone.{layer_index}.'

        if block_key in _IGNORED_BLOCK_KEYS:
            continue

        if block_key == 'attn.c_attn.weight':
            if value.ndim != 2 or value.size(1) % 3 != 0:
                raise ValueError(
                    'Expected attn.c_attn.weight to have shape '
                    '(in_features, 3 * out_features), but got '
                    f'{tuple(value.shape)}.'
                )
            q_weight, k_weight, v_weight = value.chunk(3, dim=1)
            for projection, weight in zip(
                ('q_proj', 'k_proj', 'v_proj'),
                (q_weight, k_weight, v_weight),
                strict=True,
            ):
                converted[f'{target_prefix}attn.attn.{projection}.weight'] = weight.T
            continue

        if block_key == 'attn.c_attn.bias':
            if value.ndim != 1 or value.numel() % 3 != 0:
                raise ValueError(
                    'Expected attn.c_attn.bias to have shape '
                    '(3 * out_features,), but got '
                    f'{tuple(value.shape)}.'
                )
            q_bias, k_bias, v_bias = value.chunk(3)
            for projection, bias in zip(
                ('q_proj', 'k_proj', 'v_proj'),
                (q_bias, k_bias, v_bias),
                strict=True,
            ):
                converted[f'{target_prefix}attn.attn.{projection}.bias'] = bias
            continue

        target_suffix = _BLOCK_KEYS.get(block_key)
        if target_suffix is None:
            raise KeyError(f'Unsupported GPT-2 block state-dict key: {key!r}')
        if block_key in _TRANSPOSED_BLOCK_WEIGHTS:
            value = value.T
        converted[f'{target_prefix}{target_suffix}'] = value

    return converted
