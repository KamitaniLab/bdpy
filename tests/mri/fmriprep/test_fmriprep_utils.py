"""Unit tests for fMRIPrep-related utilities in bdpy.mri.fmriprep."""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import bdpy
from bdpy import mri as fake_bdpy_mri  # type: ignore[import]

def _create_gm_flag_from_env() -> bool:
    return os.environ.get("TEST_FMRIPREP_CREATE_GOLDEN_MASTER", "0") == "1"


CREATE_GOLDEN_MASTER = _create_gm_flag_from_env()

TESTS_ROOT = Path("./tests").resolve()
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

MODULE_PATH = Path(__file__).resolve().parents[3] / "bdpy" / "mri" / "fmriprep.py"
MODULE_SPEC = importlib.util.spec_from_file_location("bdpy.mri.fmriprep", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:  # pragma: no cover - defensive
    raise ImportError("Unable to load bdpy.mri.fmriprep specification.")
fmriprep = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["bdpy.mri.fmriprep"] = fmriprep
MODULE_SPEC.loader.exec_module(fmriprep)
fake_bdpy_mri.fmriprep = fmriprep


def _get_private(name: str) -> object:
    mangled = f"_fmriprep__{name}"
    if hasattr(fmriprep, mangled):
        return getattr(fmriprep, mangled)
    return getattr(fmriprep, f"__{name}")


class RealDatasetMixin(unittest.TestCase):
    """Mixin for real-data tests."""

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

    @classmethod
    def setUpClass(cls) -> None:
        if not REAL_DATA_ROOT.exists():
            raise unittest.SkipTest(f"Real dataset not found at {REAL_DATA_ROOT}")
        if not CREATE_GOLDEN_MASTER and not REAL_EXPECTED_H5.exists():
            raise unittest.SkipTest(f"Expected reference H5 is missing: {REAL_EXPECTED_H5}")

        cls.data_root = REAL_DATA_ROOT
        cls._runtime_label_mapper_tmpdir = None
        if REAL_LABELS_MAPPER_PATH.exists():
            mapper_path = REAL_LABELS_MAPPER_PATH
        else:
            cls._runtime_label_mapper_tmpdir, mapper_path = cls._create_runtime_label_mapper(REAL_DATA_ROOT)

        cls.label_mapper = {"stimulus_name": str(mapper_path)}
        cls.expected_bdata = bdpy.BData(str(REAL_EXPECTED_H5)) if REAL_EXPECTED_H5.exists() else None
        cls.check_keys = VOLUME_NATIVE_CHECK_KEYS
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "_runtime_label_mapper_tmpdir", None) is not None:
            cls._runtime_label_mapper_tmpdir.cleanup()
        return super().tearDownClass()
