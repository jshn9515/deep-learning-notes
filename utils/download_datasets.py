import os

import torchvision.datasets as datasets

ROOT = os.getenv('DNNL_DATA_ROOT', os.path.expanduser('~/datasets'))

try:
    ds = datasets.MNIST(ROOT, download=True)
except Exception as err:
    raise ConnectionRefusedError(f'Error downloading MNIST dataset: {err}')

try:
    ds = datasets.Caltech101(ROOT, download=True)
except Exception as err:
    raise ConnectionRefusedError(f'Error downloading Caltech101 dataset: {err}')
