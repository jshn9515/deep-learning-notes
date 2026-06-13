"""Clean up .jupyter_cache directories in the current directory and its subdirectories."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    count = 0

    for folder in ['zh', 'en']:
        root = ROOT / folder

        if not root.exists():
            continue

        for nb_path in root.rglob('*.ipynb'):
            if nb_path.is_file():
                print(f'Deleting {nb_path}', flush=True)
                nb_path.unlink()
                count += 1

    print(f'Deleted {count} ipynb file(s).', flush=True)


if __name__ == '__main__':
    main()
