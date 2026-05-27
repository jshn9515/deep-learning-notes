from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    count = 0

    for mmd_path in ROOT.rglob('*.mmd'):
        svg_path = mmd_path.with_suffix('.svg')

        if svg_path.exists():
            print(f'Deleting {svg_path.name}', flush=True)
            svg_path.unlink()
            count += 1

    print(f'Deleted {count} svg file(s).', flush=True)


if __name__ == '__main__':
    main()
