"""Generate the Chinese Typst Quarto book chapter list."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / '_quarto-typst-zh.yml'
BOOK_MARKER = 'book:'
NUMBER_RE = re.compile(r'ch(?P<chapter>\d+)\.(?P<section>\d+)')

TOC = [
    'book:',
    '  chapters:',
    '    - index.qmd',
    '    - part: "Chapter 1: 深度学习简介"',
    '      chapters:',
    '        - "zh/ch1-introduction/ch1.1-neural-network.qmd"',
    '        - "zh/ch1-introduction/ch1.3-computation-graph.qmd"',
    '    - part: "Chapter 2: PyTorch 入门"',
    '      chapters:',
    '        - "zh/ch2-pytorch-introduction/*.qmd"',
    '    - part: "Chapter 3: 多层感知机：从单层到深层的非线性建模"',
    '      chapters:',
    '        - "zh/ch3-multi-layer-perceptron/*.qmd"',
    '    - part: "Chapter 4: 优化算法：神经网络如何更新参数"',
    '      chapters:',
    '        - "zh/ch4-optimization-algorithms/*.qmd"',
    '    - part: "Chapter 8: Attention 与 Transformer：从动态检索到序列建模"',
    '      chapters:',
    '        - "zh/ch8-attention-and-transformer/*.qmd"',
    '    - part: "Chapter 10: 高效 Attention 实现：从 Memory-Efficient Attention 到 FlashAttention"',
    '      chapters:',
    '        - "zh/ch10-efficient-attention/*.qmd"',
    '    - part: "Chapter 11: Vision Transformer：从图像分类到视觉序列建模"',
    '      chapters:',
    '        - "zh/ch11-vision-transformers/*.qmd"',
    '    - part: "Chapter 12: GAN：在对抗中学习生成"',
    '      chapters:',
    '        - "zh/ch12-generative-adversarial-networks/*.qmd"',
    '    - part: "Chapter 13: VAE：从压缩重建到概率生成"',
    '      chapters:',
    '        - "zh/ch13-autoencoders-and-vaes/*.qmd"',
    '    - part: "Chapter 14: Diffusion Models：从去噪到生成"',
    '      chapters:',
    '        - "zh/ch14-diffusion-models/*.qmd"',
    '    - part: "Chapter 15: 视觉语言模型：从图文对齐到多模态对话"',
    '      chapters:',
    '        - "zh/ch15-vision-language-models/*.qmd"',
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
