import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NPX = 'npx.cmd' if platform.system() == 'Windows' else 'npx'
PUPPETEER_CONFIG = ROOT / 'utils' / 'puppeteer-config.json'


def render_mmd(input_path: Path):
    output_path = input_path.with_suffix('.svg')
    print(f'Rendering {input_path.name} -> {output_path.name}', flush=True)

    subprocess.run(
        [
            NPX,
            'mmdc',
            '-i',
            str(input_path),
            '-o',
            str(output_path),
            '-p',
            str(PUPPETEER_CONFIG),
        ],
        check=True,
    )


def main():
    for input_path in ROOT.rglob('*.mmd'):
        render_mmd(input_path)


if __name__ == '__main__':
    if os.getenv('QUARTO_PROJECT_RENDER_ALL'):
        main()
