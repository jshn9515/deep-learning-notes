"""Clean up checkpoint files in the current directory and its subdirectories."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.pt', '.pth'}:
            print(f'Deleting: {p}')
            p.unlink()


if __name__ == '__main__':
    if os.getenv('QUARTO_PROJECT_RENDER_ALL'):
        main()
