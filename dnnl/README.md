# dnnl

**dnnl** is the companion Python package for **Deep Learning Notes Library**.

It provides code examples, helper functions, and small utilities used throughout the tutorial, similar in spirit to the `d2l` package for _Dive into Deep Learning_.

The package structure is similar to PyTorch, but keeps a clear boundary between reusable neural network building blocks and complete model implementations:

- `dnnl.nn` contains general neural network modules, such as attention layers, positional encodings, and other reusable components.
- `dnnl.nn.functional` contains stateless helper functions, such as functional attention implementations.
- `dnnl.models` contains higher-level model architectures or model-specific components, such as ViT, DDPM, or other models introduced in the notes.

The APIs are designed to feel close to their PyTorch counterparts where practical, while still keeping the code lightweight and easy to read for tutorial purposes.

This package is intended as a lightweight code supplement rather than a general-purpose deep learning framework. Its goal is to make the examples in the notes easier to run, reuse, and extend.

## What is this package for?

The `dnnl` package is designed to support the code in the **Deep Learning Notes Library** tutorial.

It can be used to:

- Organize example code from the notes
- Provide reusable utility functions
- Reduce repeated boilerplate in notebooks and scripts
- Make tutorial examples easier to reproduce

In short, this package serves as the code companion to the tutorial.

## Requirements

- Python 3.12 or newer
- PyTorch 3.10 or newer

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
git clone https://github.com/jshn9515/deep-learning-notes.git
cd dnnl
uv pip install .
```

If you want to modify the package while working through the notes, editable installation is recommended:

```bash
uv pip install -e .
```

This way, changes to the source code take effect immediately without reinstalling the package each time.

## Examples

After installation, you can import reusable neural network modules from `dnnl.nn`:

```python
import torch
import dnnl.nn as dnn

attn = dnn.MultiheadAttention(embed_dim=16, num_heads=4)

query = torch.randn(2, 8, 16)
key = torch.randn(2, 8, 16)
value = torch.randn(2, 8, 16)

output = attn(query, key, value)
```

You can also import stateless functions from `dnnl.nn.functional`:

```python
import torch
import dnnl.nn.functional as dF

query = torch.randn(2, 4, 8, 16)
key = torch.randn(2, 4, 8, 16)
value = torch.randn(2, 4, 8, 16)

output, weights = dF.scaled_dot_product_attention(
    query,
    key,
    value,
    need_weights=True,
)
```

Higher-level model architectures live under `dnnl.models`:

```python
import torch
from dnnl.models.vit import ViTForImageClassification

model = ViTForImageClassification(
    image_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
)

images = torch.randn(2, 3, 224, 224)
logits = model(images)
```

A simple rule of thumb is:

- Use `dnnl.nn` when a component is reusable across many models.
- Use `dnnl.models` when the code represents a complete architecture or is tightly coupled to one model family.

## License

This project is licensed under the **MIT License**.
