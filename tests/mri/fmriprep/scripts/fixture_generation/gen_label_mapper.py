"""Generate the pinned stimulus-name label mapper shipped with the real-data fixture.

``RealDatasetMixin`` in tests/mri/fmriprep/_support.py needs a mapping
from ``stimulus_name`` to an integer index. If the fixture does not provide one,
the mixin rebuilds it at run time by scanning every raw ``*_events.tsv`` in the
dataset, which would mean shipping all 560 event files rather than the six the
test actually reads.

This script produces that mapping once, from the full dataset, so it can be
pinned in the published fixture as ``golden_master/real/temp.tsv``. The ordering
rule is the same as the run-time implementation: event files are visited in
sorted path order, rows in file order, and each new ``stimulus_name`` is assigned
the next integer starting from one. Rows with a missing or ``n/a`` stimulus name
are skipped.

Because the indices depend on that traversal order, the mapper must be
regenerated from the full dataset, not from the minimal subset.

Usage:
    python gen_label_mapper.py <ds006319_root> <out_tsv>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def build_mapper(data_root):
    """Assign an integer index to every stimulus name in a BIDS dataset.

    Inputs
    ------
    data_root : pathlib.Path
        Root of the raw BIDS dataset. Files under ``derivatives`` are ignored,
        so only the raw event files contribute to the mapping.

    Output
    ------
    stimulus_names : dict of str to int
        Mapping from stimulus name to its one-based index, in order of first
        appearance.
    event_files : list of pathlib.Path
        The event files that were scanned, in the order they were visited.

    What it does
    ------------
    Collects every ``*_events.tsv`` outside ``derivatives`` in sorted path order,
    reads each row in file order, and records the first appearance of each
    stimulus name with the next available index. Missing and ``n/a`` names are
    skipped without consuming an index.
    """
    stimulus_names: dict[str, int] = {}
    event_files = sorted(
        path for path in data_root.rglob("*_events.tsv") if "derivatives" not in path.parts
    )

    for event_file in event_files:
        with event_file.open("r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                stimulus_name = row.get("stimulus_name")
                if not stimulus_name or stimulus_name == "n/a":
                    continue
                if stimulus_name not in stimulus_names:
                    stimulus_names[stimulus_name] = len(stimulus_names) + 1

    return stimulus_names, event_files


def main():
    """Build the label mapper from the command line and write it as a TSV."""
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)

    data_root = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()

    print("Building the stimulus-name label mapper")
    print("  dataset root: %s" % data_root)

    mapper, event_files = build_mapper(data_root)

    print("  event files scanned: %d" % len(event_files))
    print("  unique stimulus names: %d" % len(mapper))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for stimulus_name, index in mapper.items():
            writer.writerow([stimulus_name, index])

    print("  written to: %s" % out_path)


if __name__ == "__main__":
    main()
