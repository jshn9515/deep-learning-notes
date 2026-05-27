import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NPX = 'npx.cmd' if platform.system() == 'Windows' else 'npx'


def render_mmd(input_path: Path):
    output_path = input_path.with_suffix('.svg')
    print(f'Rendering {input_path.name} -> {output_path.name}', flush=True)

    subprocess.run(
        [NPX, 'mmdc', '-i', str(input_path), '-o', str(output_path)],
        check=True,
    )


def main():
    for input_path in ROOT.rglob('*.mmd'):
        render_mmd(input_path)


if __name__ == '__main__':
    main()
