import argparse
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NPX = 'npx.cmd' if platform.system() == 'Windows' else 'npx'
PUPPETEER_CONFIG = ROOT / 'utils' / 'puppeteer-config.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Render Mermaid .mmd files to SVG.')
    parser.add_argument(
        '-s',
        '--skip-if-exists',
        action='store_true',
        help='Skip rendering when the target SVG already exists.',
    )
    return parser.parse_args()


def render_mmd(input_path: Path) -> None:
    output_path = input_path.with_suffix('.svg')
    print(f'Rendering {input_path.name} -> {output_path.name}.', flush=True)

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


def main() -> None:
    args = parse_args()
    rendered_count = 0
    skipped_count = 0

    for input_path in ROOT.rglob('*.mmd'):
        output_path = input_path.with_suffix('.svg')

        if args.skip_if_exists and output_path.exists():
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
