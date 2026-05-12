"""Clean up .jupyter_cache directories in the current directory and its subdirectories."""

from pathlib import Path
import shutil

root = Path('.').resolve()

for cache_dir in root.rglob('.jupyter_cache'):
    if cache_dir.is_dir():
        print(f'Deleting: {cache_dir}')
        shutil.rmtree(cache_dir)
