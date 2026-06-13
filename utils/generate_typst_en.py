"""Generate the English Typst Quarto book chapter list."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / '_quarto-typst-en.yml'
BOOK_MARKER = 'book:'
NUMBER_RE = re.compile(r'ch(?P<chapter>\d+)\.(?P<section>\d+)')

TOC = [
    'book:',
    '  chapters:',
    '    - index.qmd',
    '    - part: "Chapter 1: Introduction to Deep Learning"',
    '      chapters:',
    '        - "en/ch1-introduction/*.qmd"',
    '    - part: "Chapter 2: Getting Started with PyTorch"',
    '      chapters:',
    '        - "en/ch2-pytorch-introduction/*.qmd"',
    '    - part: "Chapter 3: Multi-Layer Perceptron: From Single Layer to Deep Nonlinear Modeling"',
    '      chapters:',
    '        - "en/ch3-multi-layer-perceptron/*.qmd"',
    '    - part: "Chapter 4: Optimization Algorithms: How Neural Networks Update Parameters"',
    '      chapters:',
    '        - "en/ch4-optimization-algorithms/*.qmd"',
    '    - part: "Chapter 8: Attention and Transformer: From Dynamic Retrieval to Sequence Modeling"',
    '      chapters:',
    '        - "en/ch8-attention-and-transformer/*.qmd"',
    '    - part: "Chapter 10: Efficient Attention Implementations: From Memory-Efficient Attention to FlashAttention"',
    '      chapters:',
    '        - "en/ch10-efficient-attention/*.qmd"',
    '    - part: "Chapter 11: Vision Transformer: From Image Classification to Visual Sequence Modeling"',
    '      chapters:',
    '        - "en/ch11-vision-transformers/*.qmd"',
    '    - part: "Chapter 12: GAN: Learning to Generate through Adversarial Training"',
    '      chapters:',
    '        - "en/ch12-generative-adversarial-networks/*.qmd"',
    '    - part: "Chapter 13: VAE: From Compression and Reconstruction to Probabilistic Generation"',
    '      chapters:',
    '        - "en/ch13-autoencoders-and-vaes/*.qmd"',
    '    - part: "Chapter 14: Diffusion Models: From Denoising to Generation"',
    '      chapters:',
    '        - "en/ch14-diffusion-models/*.qmd"',
    '    - part: "Chapter 15: Vision-Language Models: From Image-Text Alignment to Multimodal Dialogue"',
    '      chapters:',
    '        - "en/ch15-vision-language-models/*.qmd"',
]


def sort_key(path: Path) -> tuple[int, int, str]:
    """Sort chapter files by numeric chapter.section when possible."""
    match = NUMBER_RE.search(path.stem)
    if match:
        return (int(match['chapter']), int(match['section']), path.name)
    return (10**9, 10**9, path.name)


def expand_toc() -> str:
    """Expand wildcard chapter entries into concrete QMD paths."""
    lines: list[str] = []

    for line in TOC:
        if '*.qmd' not in line:
            lines.append(line)
            continue

        quote = '"'
        pattern = line.split(quote)[1]
        matches = sorted(ROOT.glob(pattern), key=sort_key)
        if not matches:
            raise FileNotFoundError(f'No files matched {pattern}')

        indent = line[: len(line) - len(line.lstrip())]
        for path in matches:
            lines.append(f'{indent}- "{path.relative_to(ROOT).as_posix()}"')

    return '\n'.join(lines).rstrip() + '\n'


def write_config() -> None:
    """Replace the existing book block with the generated one."""
    config = CONFIG.read_text(encoding='utf-8').rstrip()
    marker_index = config.find(BOOK_MARKER)
    if marker_index == -1:
        next_config = f'{config}\n\n{expand_toc()}'
    else:
        next_config = f'{config[:marker_index].rstrip()}\n\n{expand_toc()}'

    CONFIG.write_text(next_config, encoding='utf-8')


def main() -> None:
    write_config()
    print(f'Updated {CONFIG.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
