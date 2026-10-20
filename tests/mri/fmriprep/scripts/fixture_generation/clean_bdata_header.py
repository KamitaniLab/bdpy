"""Strip the BData ``header`` group from a golden-master ``.h5`` before publishing it.

``BData.save()`` records provenance for every save: it walks the call stack and
writes each frame's file path and source code into ``/header/callstack`` and
``/header/callstack_code``. For the real-data golden master this embeds absolute
paths from the machine that generated it, including the user name and the host
name, which must not appear in a published dataset.

The script copies every top-level group except ``header`` into a fresh file.
Deleting the group in place is not sufficient: HDF5 unlinks it but leaves the
raw bytes in freed space, so the paths remain recoverable from the file. Writing
a new file guarantees they are gone.

Dropping ``header`` does not affect the test, which compares only ``dataset``,
``metadata`` and ``vmap``.

Usage:
    python clean_bdata_header.py <in.h5> [out.h5]   # defaults to in-place
"""
import shutil
import sys
import tempfile
from pathlib import Path

import h5py

DROP_GROUPS = {"header"}


def strip_header(src, dst):
    """Copy an HDF5 file while omitting the provenance groups.

    Inputs
    ------
    src : pathlib.Path
        Path to the source ``.h5`` file, read only.
    dst : pathlib.Path
        Path the cleaned file is written to. May be the same as ``src``, in
        which case the file is replaced once the copy has been completed.

    Output
    ------
    dropped : list of str
        Names of the top-level groups that were omitted. Empty if the file did
        not contain any of them.

    What it does
    ------------
    Writes a temporary file next to ``dst``, copies the root attributes and
    every top-level group except those listed in ``DROP_GROUPS`` into it, and
    then moves the temporary file over ``dst``.
    """
    tmp_path = Path(tempfile.mkstemp(suffix=".h5", dir=str(dst.parent))[1])
    dropped = []

    with h5py.File(src, "r") as fin, h5py.File(tmp_path, "w") as fout:
        for key, value in fin.attrs.items():
            fout.attrs[key] = value
        for key in fin.keys():
            if key in DROP_GROUPS:
                dropped.append(key)
                continue
            fin.copy(key, fout)

    shutil.move(str(tmp_path), str(dst))
    return dropped


def main():
    """Run the header stripping from the command line."""
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        raise SystemExit(2)

    src = Path(sys.argv[1]).resolve()
    dst = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else src

    print("Stripping BData provenance header")
    print("  input : %s" % src)
    print("  output: %s" % dst)

    dropped = strip_header(src, dst)

    if dropped:
        print("  dropped top-level groups: %s" % ", ".join(dropped))
    else:
        print("  nothing to strip; the file had no header group")


if __name__ == "__main__":
    main()
