# dnnl

**dnnl** is the companion Python package for **Deep Learning Notes Library**.

It provides code examples, helper functions, and small utilities used throughout the tutorial, similar in spirit to the `d2l` package for _Dive into Deep Learning_.

The package structure is similar to PyTorch: module classes live under `dnnl.nn`, and stateless helper functions live under `dnnl.nn.functional`. The APIs are also designed to feel close to their PyTorch counterparts where practical.

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

## Example

After installation, you can import neural network modules from `dnnl.nn`:

```python
import torch
import dnnl.nn as nn

attn = nn.MultiheadAttention(embed_dim=16, num_heads=4)

query = torch.randn(2, 8, 16)
key = torch.randn(2, 8, 16)
value = torch.randn(2, 8, 16)

output = attn(query, key, value)
```

You can also import stateless functions from `dnnl.nn.functional`:

```python
import torch
import dnnl.nn.functional as F

query = torch.randn(2, 4, 8, 16)
key = torch.randn(2, 4, 8, 16)
value = torch.randn(2, 4, 8, 16)

output, weights = F.scaled_dot_product_attention(
    query,
    key,
    value,
    need_weights=True,
)
```

## License

This project is licensed under the **MIT License**.
