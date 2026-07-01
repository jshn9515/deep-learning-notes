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
    rendered_count = 0
    skipped_count = 0

    for input_path in ROOT.rglob('*.mmd'):
        output_path = input_path.with_suffix('.svg')

        if output_path.exists():
            print(f'Skipping {input_path.name} as it already exists.', flush=True)
            skipped_count += 1
            continue

        render_mmd(input_path)
        rendered_count += 1

    print(
        f'Rendered {rendered_count} mmd file(s); skipped {skipped_count} existing svg file(s).',
        flush=True,
    )


if __name__ == '__main__':
    main()
