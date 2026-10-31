"""dnnlpy: A utility library for deep learning notes, providing common functions and tools
to support the main content of the deep learning notes collection.
"""

from functools import cache

from .configtools import (
    get_data_dir as get_data_dir,
    get_data_path as get_data_path,
    get_default_device as get_default_device,
    get_experiment_dir as get_experiment_dir,
    get_experiment_path as get_experiment_path,
    get_num_workers as get_num_workers,
    has_gil as has_gil,
    set_seed as set_seed,
)
from .pylabtools import set_matplotlib_format as set_matplotlib_format
from .trainingtools import Trainer as Trainer
from .utils import (
    bytes_to_gib as bytes_to_gib,
    bytes_to_mib as bytes_to_mib,
    count_params as count_params,
)


@cache
def is_avx2_vnni_2() -> bool:
    """Check if the current CPU supports AVX2, VNNI, and related features.

    See https://github.com/pytorch/pytorch/issues/148861 for more details.
    Code ported from https://github.com/intel/torch-xpu-ops/pull/4341.
    """
    from torch.cpu import get_capabilities

    cap = get_capabilities()
    return (
        cap.get('avx2', False)
        and cap.get('avx_vnni', False)
        and cap.get('avx_vnni_int8', False)
        and cap.get('avx_ne_convert', False)
        and not cap.get('avx512_f', False)
    )
