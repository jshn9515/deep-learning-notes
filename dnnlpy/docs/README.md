# Build the `dnnlpy` API reference

The reference uses Sphinx autodoc and PyData Sphinx Theme. It reads signatures and docstrings from the local `dnnlpy/src` tree. Build it in a Python environment where `dnnlpy` and its runtime dependencies can be imported. The documentation-only dependencies are listed in `requirements.txt`.

From the repository root:

```bash
uv venv .venv --python 3.14
uv pip install --python .venv -e dnnlpy -r dnnlpy/docs/requirements.txt
.venv/bin/sphinx-build -b html dnnlpy/docs _site/dnnlpy-api
```

On Windows, use `.venv/Scripts/sphinx-build.exe` in the last command. Open `_site/dnnlpy-api/index.html` to view the result.

The `reference/*.rst` pages are Sphinx sources. When package modules are added or removed, regenerate them with:

```bash
sphinx-apidoc --separate --force -o dnnlpy/docs/reference dnnlpy/src/dnnlpy
```

Review the generated files before committing them. Ensure the referenced modules are included in the same checkout as the documentation.
