from dataclasses import dataclass


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
