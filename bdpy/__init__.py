"""BdPy: Brain decoding toolbox for Python.

Developed by Kamitani Lab, Kyoto Univ. and ATR.
"""


# `import bdpy` implicitly imports class `BData` (in package `bdata`) and
# package `util`.
from .bdata import BData, metadata_equal, vstack
from .util import (
    average_elemwise,
    create_groupvector,
    divide_chunks,
    dump_info,
    get_refdata,
    makedir_ifnot,
)
