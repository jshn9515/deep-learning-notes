# dnnl

**dnnl** is the companion Python package for **Deep Learning Notes Library**.

It provides code examples, helper functions, and small utilities used throughout the tutorial, similar in spirit to the `d2l` package for _Dive into Deep Learning_.

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

After installation, you can import the package in Python:

```python
import dnnl.<chapter No.>
```

For example, the code supplement for Chapter 1 is placed in `ch1`, and the code supplement for Chapter 2 is placed in `ch2`.

As more chapters are added to the notes, the package will continue to grow with corresponding chapter-based modules and utilities.

## License

This project is licensed under the **MIT License**.
