"""Shared helpers for the bdpy.mri.fmriprep tests.

This module holds no tests; the leading underscore keeps pytest from
collecting it.
"""

from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

import bdpy
from bdpy.mri import fmriprep

def _create_gm_flag_from_env() -> bool:
    return os.environ.get("TEST_FMRIPREP_CREATE_GOLDEN_MASTER", "0") == "1"


CREATE_GOLDEN_MASTER = _create_gm_flag_from_env()

#: Anchored to this file rather than to the working directory, so the tests
#: pass no matter where pytest is invoked from.
TESTS_ROOT = Path(__file__).resolve().parents[2]
MOCK_GOLDEN_MASTER_DIR = TESTS_ROOT / "data" / "mri" / "golden_master" / "mock"
REAL_EXPECTED_H5 = TESTS_ROOT / "data" / "mri" / "golden_master" / "real" / "test_output_fmriprep_real_exclude.h5"
REAL_LABELS_MAPPER_PATH = TESTS_ROOT / "data" / "mri" / "golden_master" / "real" / "temp.tsv"


def _resolve_real_data_root() -> Path:
    candidates = (
        TESTS_ROOT / "data" / "mri" / "ds006319",
        TESTS_ROOT / "data" / "ds006319",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


REAL_DATA_ROOT = _resolve_real_data_root()

VOLUME_NATIVE_CHECK_KEYS = [
    "VoxelData",
    "Session",
    "Run",
    "Block",
    "Label",
    "MotionParameter",
    "MotionParameter_trans_x",
    "MotionParameter_trans_y",
    "MotionParameter_trans_z",
    "MotionParameter_rot_x",
    "MotionParameter_rot_y",
    "MotionParameter_rot_z",
    "Confounds",
    "GlobalSignal",
    "WhiteMatterSignal",
    "CSFSignal",
    "DVARS",
    "STD_DVARS",
    "FramewiseDisplacement",
    "aCompCor",
    "aCompCor_0",
    "aCompCor_1",
    "aCompCor_2",
    "aCompCor_3",
    "aCompCor_4",
    "aCompCor_5",
    "tCompCor",
    "tCompCor_0",
    "tCompCor_1",
    "tCompCor_2",
    "tCompCor_3",
    "tCompCor_4",
    "tCompCor_5",
    "Cosine",
    "Cosine_0",
    "Cosine_1",
    "Cosine_2",
    "Cosine_3",
    "Cosine_4",
    "Cosine_5",
    "trial_type",
    "stimulus_name",
    "category_index",
    "response_time",
    "voxel_x",
    "voxel_y",
    "voxel_z",
    "voxel_i",
    "voxel_j",
    "voxel_k",
]

def _get_private(name: str) -> object:
    mangled = f"_fmriprep__{name}"
    if hasattr(fmriprep, mangled):
        return getattr(fmriprep, mangled)
    return getattr(fmriprep, f"__{name}")


class RealDatasetMixin(unittest.TestCase):
    """Mixin for real-data tests."""

    #: Whether the 2.5 GB stored-expectation h5 is required. Tests that assert
    #: on the dataset itself rather than on a recorded result set this False so
    #: they run with only the 478 MB fixture present.
    requires_golden_master: bool = True

    @staticmethod
    def _create_runtime_label_mapper(data_root: Path) -> tuple[tempfile.TemporaryDirectory, Path]:
        tmpdir = tempfile.TemporaryDirectory(prefix="test_fmriprep_sound_label_mapper_")
        mapper_path = Path(tmpdir.name) / "stimulus_name.tsv"

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

        with mapper_path.open("w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for stimulus_name, index in stimulus_names.items():
                writer.writerow([stimulus_name, index])

        return tmpdir, mapper_path

    #: Shown whenever the real-data fixture is missing.
    _DOWNLOAD_HINT = (
        "Fetch it first:\n"
        "    bash ./tests/mri/fmriprep/scripts/real/step_1_figshare_download.sh\n"
        "The download is deliberately explicit: it pulls ~1.7 GB from figshare, "
        "so it never runs as a side effect of pytest."
    )

    @classmethod
    def setUpClass(cls) -> None:
        # tests/conftest.py deselects these unless the marker expression names
        # real_data, so reaching this point means the caller asked for them by
        # name. Skipping would let that request pass silently as a success, so a
        # missing fixture is an error instead.
        if not REAL_DATA_ROOT.exists():
            raise FileNotFoundError(
                f"Real-data fixture not found: {REAL_DATA_ROOT}\n{cls._DOWNLOAD_HINT}"
            )
        if (
            cls.requires_golden_master
            and not CREATE_GOLDEN_MASTER
            and not REAL_EXPECTED_H5.exists()
        ):
            raise FileNotFoundError(
                f"Real-data golden master not found: {REAL_EXPECTED_H5}\n"
                f"{cls._DOWNLOAD_HINT}\n"
                "To regenerate it instead, set TEST_FMRIPREP_CREATE_GOLDEN_MASTER=1."
            )

        cls.data_root = REAL_DATA_ROOT
        cls._runtime_label_mapper_tmpdir = None
        if REAL_LABELS_MAPPER_PATH.exists():
            mapper_path = REAL_LABELS_MAPPER_PATH
        else:
            cls._runtime_label_mapper_tmpdir, mapper_path = cls._create_runtime_label_mapper(REAL_DATA_ROOT)

        cls.label_mapper = {"stimulus_name": str(mapper_path)}
        # Reading the h5 costs minutes and ~2.5 GB, so only subclasses that
        # actually compare against it pay for it.
        if cls.requires_golden_master and REAL_EXPECTED_H5.exists():
            cls.expected_bdata = bdpy.BData(str(REAL_EXPECTED_H5))
        else:
            cls.expected_bdata = None
        cls.check_keys = VOLUME_NATIVE_CHECK_KEYS
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "_runtime_label_mapper_tmpdir", None) is not None:
            cls._runtime_label_mapper_tmpdir.cleanup()
        return super().tearDownClass()
