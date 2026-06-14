"""Clean up checkpoint files in the current directory and its subdirectories."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.pt', '.pth'}:
            print(f'Deleting: {p}', flush=True)
            p.unlink()


if __name__ == '__main__':
    main()
