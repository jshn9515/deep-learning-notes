from dataclasses import dataclass

__all__ = [
    'GPT2_LARGE',
    'GPT2_MEDIUM',
    'GPT2_SMALL',
    'GPT2_TINY',
    'GPT2_XL',
    'GPT2Config',
]


@dataclass
class GPT2Config:
    """Configuration for the Transformer model.

    Args:
        vocab_size (int): Size of the vocabulary.
        context_length (int): Maximum length of the input sequence.
        d_model (int): Dimensionality of the model's hidden states.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of layers in the Transformer model.
        weight_tying (bool, default: True): Whether to tie the weights of the
            input and output embeddings.
        fast (bool, optional): Whether to use fast attention implementation.
    """

    vocab_size: int
    context_length: int
    d_model: int
    n_heads: int
    n_layers: int
    weight_tying: bool = True
    fast: bool = False


GPT2_TINY = GPT2Config(
    vocab_size=50257,
    context_length=512,
    d_model=256,
    n_heads=4,
    n_layers=4,
)

GPT2_SMALL = GPT2Config(
    vocab_size=50257,
    context_length=1024,
    d_model=768,
    n_heads=12,
    n_layers=12,
)

GPT2_MEDIUM = GPT2Config(
    vocab_size=50257,
    context_length=1024,
    d_model=1024,
    n_heads=16,
    n_layers=24,
)

GPT2_LARGE = GPT2Config(
    vocab_size=50257,
    context_length=1024,
    d_model=1280,
    n_heads=20,
    n_layers=36,
)

GPT2_XL = GPT2Config(
    vocab_size=50257,
    context_length=1024,
    d_model=1600,
    n_heads=25,
    n_layers=48,
)
