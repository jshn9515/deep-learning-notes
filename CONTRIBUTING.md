# Contributing

Thank you for your interest in contributing to this project.

To keep the repository consistent and easy to maintain, please follow the steps below before submitting a Pull Request.

## 1. Install Quarto

This project is maintained primarily in **Quarto Markdown (`.qmd`)** format.

Please install Quarto first from the official website:

- Quarto Get Started: <https://quarto.org/docs/get-started/>

After installation, make sure the `quarto` command is available in your terminal.

## 2. Install `dnnl`

Before running or modifying the notes, please enter the `dnnl` directory and install the `dnnl` library according to the instructions in `dnnl/README.md`.

Some notebooks and examples rely on utilities and custom implementations provided by `dnnl`, so skipping this step may cause rendering or execution issues.

## 3. Edit the `.qmd` files

Please make your changes by editing the corresponding **Quarto Markdown (`.qmd`)** files.

When contributing:

- Follow existing `.qmd` writing style and project structure
- Keep explanations clear and concise
- If you add code examples, make sure they are readable and properly formatted
- If you modify formulas, derivations, or technical explanations, please check them carefully for correctness

## 4. Re-render locally before submitting

Before opening a Pull Request, you **must** re-render the modified content locally and confirm that everything works correctly.

At minimum, please make sure:

- The page renders successfully with no build errors
- Code blocks and math formulas display correctly
- Links, formatting, and section structure look normal
- If needed, the `.qmd` file can still be converted into a notebook with Quarto

For example, you may use:

```bash
quarto render
```

If you need to convert a `.qmd` file into a notebook for checking, you can also use:

```bash
quarto convert path/to/file.qmd
```

Please do **not** submit a PR without verifying the local rendering result first.

## 5. Submit a Pull Request

Once everything looks good locally, you can submit a Pull Request.

If your change is relatively large, it is recommended to open an Issue first to briefly describe your idea before starting implementation.

Typical contributions include:

- Fixing errors or unclear explanations
- Improving code examples or comments
- Correcting formatting or structure issues
- Adding better derivations or clearer technical explanations
- Suggesting or contributing new topics

Thank you for helping improve these notes.
