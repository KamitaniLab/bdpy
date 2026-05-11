"""Mock-data tests for bdpy.mri.fmriprep (fmriprep version 1.2)."""

from __future__ import annotations

import itertools
import unittest
from pathlib import Path
from typing import Optional
import tempfile
import os

import nibabel as nib
import numpy as np
from nipy.io.nifti_ref import NiftiError
from nibabel.gifti import GiftiDataArray, GiftiImage

import bdpy
from .test_fmriprep_utils_mock import (
    MockBidsBuilder,
    build_expected_bdata_after_exclude,
)
from .test_fmriprep_utils import (
    CREATE_GOLDEN_MASTER,
    _get_private,
    fmriprep,
)

DATA_BUILDER = MockBidsBuilder()
DATA_BUILDER.build()


class MockDatasetMixin(unittest.TestCase):
    """Mixin providing access to the shared mock dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the mock dataset for tests."""
        cls.data_root = DATA_BUILDER.root
        cls.subject = DATA_BUILDER.subject
        cls.label_mapper = {"stimulus_name": str(DATA_BUILDER.label_mapper_path)} if DATA_BUILDER.label_mapper_path is not None else None
        return super().setUpClass()


# -----------------------------------------------------------------------------
# Mock tests (default)
# -----------------------------------------------------------------------------


class TestFmriprepDataMock(MockDatasetMixin):
    """Tests for parsing mock fMRIPrep dataset structure."""

    def test_fmriprep_private_variables(self) -> None:
        """Verify internal configuration fields are initialized."""
        instance = fmriprep.FmriprepData(self.data_root.as_posix())
        self.assertEqual(instance._FmriprepData__datapath, self.data_root.as_posix())
        self.assertEqual(instance._FmriprepData__fmriprep_dir, "derivatives/fmriprep")
        self.assertEqual(instance._FmriprepData__fmriprep_version, "1.2")

    def test_fmriprep_get_subjects(self) -> None:
        """Verify subject discovery returns the mock subject."""
        instance = fmriprep.FmriprepData(self.data_root.as_posix())
        prepdir = (self.data_root / "derivatives" / "fmriprep" / "fmriprep").as_posix()
        subjects = instance._FmriprepData__get_subjects(prepdir)
        self.assertEqual(subjects, [self.subject])

    def test_fmriprep_get_sessions(self) -> None:
        """Verify session discovery returns the expected sessions."""
        instance = fmriprep.FmriprepData(self.data_root.as_posix())
        prepdir = (self.data_root / "derivatives" / "fmriprep" / "fmriprep").as_posix()
        sessions = instance._FmriprepData__get_sessions(prepdir, self.subject)
        self.assertEqual(sessions, ["ses-01", "ses-02", "ses-anat"])

    def test_fmriprep_parse_session(self) -> None:
        """Verify parsing a session yields the expected runs."""
        prepdir = (self.data_root / "derivatives" / "fmriprep" / "fmriprep").as_posix()
        instance = fmriprep.FmriprepData(self.data_root.as_posix())
        runs = instance._FmriprepData__parse_session(prepdir, self.subject, "ses-01")
        self.assertEqual(runs, DATA_BUILDER.expected_runs["ses-01"])

    def test_fmriprep_parse_data(self) -> None:
        """Verify full dataset parsing matches the mock fixture."""
        instance = fmriprep.FmriprepData(self.data_root.as_posix())
        instance._FmriprepData__parse_data()
        self.assertEqual(instance._FmriprepData__data, DATA_BUILDER.expected_subject_data)

    def test_fmriprep_get_task_event_files(self) -> None:
        """Verify task event and JSON sidecars are attached to runs."""
        instance = fmriprep.FmriprepData(self.data_root.as_posix())
        instance._FmriprepData__get_task_event_files()
        for runs in instance.data[self.subject].values():
            for run in runs:
                self.assertIn("task_event_file", run)
                self.assertIn("bold_json", run)
                self.assertTrue((self.data_root / run["task_event_file"]).exists())
                self.assertTrue((self.data_root / run["bold_json"]).exists())

class TestFmriprepDataFailures(unittest.TestCase):
    """Failure-path tests for invalid mock dataset inputs."""

    def setUp(self) -> None:
        """Build an isolated mock dataset for failure scenarios."""
        self.builder = MockBidsBuilder()
        self.builder.build()
        self.label_mapper = {"stimulus_name": str(self.builder.label_mapper_path)} if self.builder.label_mapper_path is not None else None
        self.addCleanup(self.builder.cleanup)

    def test_missing_event_file_raises(self) -> None:
        """Raise when a task event file is missing."""
        event_path = Path(next(iter(self.builder.events_map.keys())))
        event_path.unlink()
        with self.assertRaisesRegex(RuntimeError, "task event files"):
            fmriprep.FmriprepData(self.builder.root.as_posix())

    def test_missing_bold_json_raises(self) -> None:
        """Raise when a BOLD JSON sidecar is missing."""
        bold_path = Path(next(iter(self.builder.bold_json_map.keys())))
        bold_path.unlink()
        with self.assertRaisesRegex(RuntimeError, "bold parameter json"):
            fmriprep.FmriprepData(self.builder.root.as_posix())

    def test_missing_confounds_file_raises(self) -> None:
        """Raise when a confounds file is missing."""
        conf_path = Path(next(iter(self.builder.confounds_map.keys())))
        conf_path.unlink()
        with self.assertRaises(TypeError):
            fmriprep.create_bdata_fmriprep(
                dpath=self.builder.root.as_posix(),
                data_mode="volume_native",
                fmriprep_dir="derivatives/fmriprep",
                label_mapper=self.label_mapper,
                with_confounds=True,
            )

    def test_confounds_missing_motion_columns_raise(self) -> None:
        """Raise when required motion columns are absent."""
        conf_path = Path(next(iter(self.builder.confounds_map.keys())))
        conf_df = self.builder.confounds_map[str(conf_path)].drop(columns=["RotZ"])
        conf_df.to_csv(conf_path, sep="\t", index=False)
        with self.assertRaisesRegex(RuntimeError, "Invalid confounds file"):
            fmriprep.create_bdata_fmriprep(
                dpath=self.builder.root.as_posix(),
                data_mode="volume_native",
                fmriprep_dir="derivatives/fmriprep",
                label_mapper=self.label_mapper,
                with_confounds=True,
            )

    def test_unknown_label_in_events_propagates_error(self) -> None:
        """Propagate label-mapper errors from event contents."""
        event_path = Path(next(iter(self.builder.events_map.keys())))
        events_df = self.builder.events_map[str(event_path)].copy()
        events_df.loc[0, "stimulus_name"] = "unknown_label"
        events_df.to_csv(event_path, sep="\t", index=False)
        with self.assertRaises(KeyError):
            fmriprep.create_bdata_fmriprep(
                dpath=self.builder.root.as_posix(),
                data_mode="volume_native",
                fmriprep_dir="derivatives/fmriprep",
                label_mapper=self.label_mapper,
                with_confounds=False,
            )

class TestCreateBdataFmriprepMock(MockDatasetMixin):
    """Tests for creating BData objects from mock inputs."""

    def setUp(self) -> None:
        """Prepare the metadata keys compared in golden-master tests."""
        self.check_keys = [
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
            "image_index",
            "response_time",
            "original_run_number",
            "voxel_x",
            "voxel_y",
            "voxel_z",
            "voxel_i",
            "voxel_j",
            "voxel_k",
        ]
        # Surface keys: VoxelData -> VertexData, voxel_x/y/z/i/j/k -> vertex_index +
        # the VertexLeft/VertexRight indicator metadata. The remaining keys
        # (Session/Run/Block/Label/MotionParameter/Confounds/label submeta) are
        # identical because production handles them identically for volume and
        # surface modes.
        self.surface_check_keys = [
            "VertexData",
            "VertexLeft",
            "VertexRight",
            "vertex_index",
        ] + [
            k for k in self.check_keys
            if k not in {"VoxelData", "voxel_x", "voxel_y", "voxel_z", "voxel_i", "voxel_j", "voxel_k"}
        ]

    def _num_voxels(self) -> int:
        """Return the voxel count in the mock volume."""
        return int(np.prod(DATA_BUILDER.voxel_shape))

    # helper to build expected BData after applying exclude
    def _expected_after_exclude(
        self, exclude: Optional[dict] = None, data_mode: str = "volume_native"
    ) -> bdpy.BData:
        """Build the expected BData after applying exclusions."""
        return build_expected_bdata_after_exclude(
            DATA_BUILDER,
            {"stimulus_name": {"face": 1, "scene": 2}},
            exclude=exclude,
            data_mode=data_mode,
        )
    
    # helper to get expected sample count after applying exclude
    def _expected_sample_count(self, exclude: Optional[dict] = None) -> int:
        """Return the expected sample count after exclusions."""
        return int(self._expected_after_exclude(exclude).get("VoxelData").shape[0])
    
    def test_create_bdata_fmriprep_gm(self) -> None:
        """Match the generated mock BData against the golden master."""
        '''
        CAUTION: 
        This test uses golden master files. 
        If you need to add some new test cases, please generate new golden master files
        '''
        save_path = "./tests/data/mri/golden_master/mock/test_output_fmriprep_subject.h5"
        # -------------------------------------------------------------
        # FOR GOLDEN MASTER UPDATE ONLY:
        if CREATE_GOLDEN_MASTER:
            expected_bdata = self._expected_after_exclude()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            expected_bdata.save(save_path)
        # -------------------------------------------------------------
        
        expected_bdata = bdpy.BData(str(save_path))
        bdata_list, data_labels_list = fmriprep.create_bdata_fmriprep(
            dpath=self.data_root.as_posix(),
            data_mode="volume_native",
            fmriprep_dir="derivatives/fmriprep",
            label_mapper=self.label_mapper,
            split_task_label=False,
            with_confounds=True,
            return_data_labels=True,
            return_list=True,
        )
        self.assertEqual(data_labels_list, [self.subject])
        self.assertEqual(len(bdata_list), 1)
        brain_data = bdata_list[0]
        for key in self.check_keys:
            np.testing.assert_array_equal(brain_data.get(key), expected_bdata.get(key))

    def test_create_bdata_fmriprep_exclude_gm(self) -> None:
        """Match excluded mock BData against the golden master."""
        '''
        CAUTION: 
        This test uses golden master files. 
        If you need to add some new test cases, please generate new golden master files
        '''
        
        exclude_list = [{"subject": ["sub-4649"], "session/run": [[1, 2], None]}]
        for exclude in exclude_list:
            save_path = "./tests/data/mri/golden_master/mock/test_output_fmriprep_subject_exclude.h5"
            # -------------------------------------------------------------
            # FOR GOLDEN MASTER UPDATE ONLY:
            if CREATE_GOLDEN_MASTER:
                expected_bdata = self._expected_after_exclude(exclude)
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                expected_bdata.save(save_path)
            # -------------------------------------------------------------
            
            expected_bdata = bdpy.BData(str(save_path))
            bdata_list, data_labels_list = fmriprep.create_bdata_fmriprep(
                dpath=self.data_root.as_posix(),
                data_mode="volume_native",
                fmriprep_dir="derivatives/fmriprep",
                label_mapper=self.label_mapper,
                exclude=exclude,
                split_task_label=False,
                with_confounds=True,
                return_data_labels=True,
                return_list=True,
            )
            self.assertEqual(data_labels_list, [self.subject])
            self.assertEqual(len(bdata_list), 1)
            brain_data = bdata_list[0]
            for key in self.check_keys:
                np.testing.assert_array_equal(brain_data.get(key), expected_bdata.get(key))

    def test_create_bdata_fmriprep_surface_native_gm(self) -> None:
        """Match the surface_native BData against the golden master.

        This test uses a golden-master file; to add or change coverage,
        regenerate it via ``TEST_FMRIPREP_CREATE_GOLDEN_MASTER=1``.
        """
        exclude = {"session/run": [[1, 2], None]}
        save_path = "./tests/data/mri/golden_master/mock/test_output_fmriprep_subject_surface.h5"
        # -------------------------------------------------------------
        # FOR GOLDEN MASTER UPDATE ONLY:
        if CREATE_GOLDEN_MASTER:
            expected_bdata = self._expected_after_exclude(exclude, data_mode="surface_native")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            expected_bdata.save(save_path)
        # -------------------------------------------------------------

        expected_bdata = bdpy.BData(str(save_path))
        bdata_list, data_labels_list = fmriprep.create_bdata_fmriprep(
            dpath=self.data_root.as_posix(),
            data_mode="surface_native",
            fmriprep_dir="derivatives/fmriprep",
            label_mapper=self.label_mapper,
            exclude=exclude,
            split_task_label=False,
            with_confounds=True,
            return_data_labels=True,
            return_list=True,
        )
        self.assertEqual(data_labels_list, [self.subject])
        self.assertEqual(len(bdata_list), 1)
        brain_data = bdata_list[0]
        for key in self.surface_check_keys:
            np.testing.assert_array_equal(brain_data.get(key), expected_bdata.get(key))

    def test_create_bdata_fmriprep_split_task_label_single_task(self) -> None:
        """split_task_label=True exercises the per-task split branch.

        The mock dataset uses a single task label (``task-mock``), so this test
        verifies that:
        (a) data_labels carry the ``<subject>_<task>`` suffix produced by the
            split branch (bdpy/mri/fmriprep.py L331-L340);
        (b) the returned BData is content-identical to the non-split golden
            master because there is only one task to split on.

        Multi-task split (where bdata_list has more than one element) is
        covered by test_fmriprep_real.py with actual fMRIPrep outputs.
        """
        save_path = "./tests/data/mri/golden_master/mock/test_output_fmriprep_subject.h5"
        expected_bdata = bdpy.BData(str(save_path))
        bdata_list, data_labels_list = fmriprep.create_bdata_fmriprep(
            dpath=self.data_root.as_posix(),
            data_mode="volume_native",
            fmriprep_dir="derivatives/fmriprep",
            label_mapper=self.label_mapper,
            split_task_label=True,
            with_confounds=True,
            return_data_labels=True,
            return_list=True,
        )
        self.assertEqual(data_labels_list, [f"{self.subject}_task-mock"])
        self.assertEqual(len(bdata_list), 1)
        brain_data = bdata_list[0]
        for key in self.check_keys:
            np.testing.assert_array_equal(brain_data.get(key), expected_bdata.get(key))

    def test_create_bdata_fmriprep_surface_native(self) -> None:
        """Create native-surface data with the expected shape."""
        exclude = {"session/run": [[1, 2], None]}
        bdata_list, _ = fmriprep.create_bdata_fmriprep(
            dpath=self.data_root.as_posix(),
            data_mode="surface_native",
            fmriprep_dir="derivatives/fmriprep",
            label_mapper=self.label_mapper,
            exclude=exclude,
            split_task_label=False,
            with_confounds=False,
            return_data_labels=True,
            return_list=True,
        )
        brain_data = bdata_list[0]
        self.assertEqual(
            brain_data.get("VertexData").shape,
            (self._expected_sample_count(exclude), 8),
        )

    def test_create_bdata_surface_resolution_variants(self) -> None:
        """Create standard-surface variants with the expected shape."""
        exclude = {"session/run": [[1, 2], None]}
        for mode in ["surface_standard", "surface_standard_41k", "surface_standard_10k"]:
            with self.subTest(mode=mode):
                bdata_list, _ = fmriprep.create_bdata_fmriprep(
                    dpath=self.data_root.as_posix(),
                    data_mode=mode,
                    fmriprep_dir="derivatives/fmriprep",
                    label_mapper=self.label_mapper,
                    exclude=exclude,
                    split_task_label=False,
                    with_confounds=False,
                    return_data_labels=True,
                    return_list=True,
                )
                self.assertEqual(
                    bdata_list[0].get("VertexData").shape,
                    (self._expected_sample_count(exclude), 8),
                )

    def test_create_bdata_fmriprep_exclude_subject_removes_all(self) -> None:
        """Return no outputs when the only subject is excluded."""
        bdata_list, data_labels_list = fmriprep.create_bdata_fmriprep(
            dpath=self.data_root.as_posix(),
            data_mode="volume_native",
            fmriprep_dir="derivatives/fmriprep",
            label_mapper=self.label_mapper,
            exclude={"subject": [self.subject]},
            split_task_label=False,
            with_confounds=False,
            return_data_labels=True,
            return_list=True,
        )
        self.assertEqual(bdata_list, [])
        self.assertEqual(data_labels_list, [])

class TestBrainDataMock(unittest.TestCase):
    """Tests for BrainData helpers using mock files."""

    @classmethod
    def setUpClass(cls) -> None:
        """Capture shared surface paths from the mock dataset."""
        cls.data_root = DATA_BUILDER.root
        cls.surface_run = DATA_BUILDER.expected_runs["ses-01"][0]["surface_native"]
        return super().setUpClass()

    def _write_gifti(self, path: Path, arrays: list[np.ndarray]) -> None:
        """Write a temporary GIFTI file from array payloads."""
        darrays = [GiftiDataArray(arr.astype(np.float32)) for arr in arrays]
        img = GiftiImage(darrays=darrays)
        nib.save(img, path)

    def test_load_surface_from_gifti_pairs(self) -> None:
        """Load paired GIFTI hemispheres into a single surface matrix."""
        left_path = self.data_root / self.surface_run[0]
        right_path = self.data_root / self.surface_run[1]
        left_img = nib.load(left_path)
        right_img = nib.load(right_path)
        brain = fmriprep.BrainData((left_path.as_posix(), right_path.as_posix()), dtype="surface")
        expected_left = np.vstack([da.data for da in left_img.darrays])
        expected_right = np.vstack([da.data for da in right_img.darrays])
        expected = np.hstack([expected_left, expected_right])
        np.testing.assert_array_equal(brain.data, expected)
        self.assertEqual(brain.index.shape, (1, expected.shape[1]))
        self.assertEqual(brain.n_vertex, (expected_left.shape[1], expected_right.shape[1]))
        with self.assertRaises(NotImplementedError):
            _ = brain.xyz

    def test_load_surface_missing_hemi_raises(self) -> None:
        """Raise when one hemisphere file is missing."""
        left_path = self.data_root / self.surface_run[0]
        missing_right = self.data_root / "missing_right.func.gii"
        with self.assertRaises(FileNotFoundError):
            fmriprep.BrainData((left_path.as_posix(), missing_right.as_posix()), dtype="surface")

    def test_load_surface_empty_darrays(self) -> None:
        """Raise when both hemisphere files contain no data arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = Path(tmpdir) / "lh_empty.func.gii"
            right_path = Path(tmpdir) / "rh_empty.func.gii"
            self._write_gifti(left_path, [])
            self._write_gifti(right_path, [])
            with self.assertRaises(ValueError):
                fmriprep.BrainData((left_path.as_posix(), right_path.as_posix()), dtype="surface")

    def test_load_surface_vertex_mismatch(self) -> None:
        """Allow mismatched hemisphere vertex counts and concatenate data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = Path(tmpdir) / "lh.func.gii"
            right_path = Path(tmpdir) / "rh.func.gii"
            self._write_gifti(left_path, [np.array([1.0, 2.0])])
            self._write_gifti(right_path, [np.array([5.0, 6.0, 7.0, 8.0])])
            brain = fmriprep.BrainData((left_path.as_posix(), right_path.as_posix()), dtype="surface")
            self.assertEqual(brain.data.shape, (1, 6))
            self.assertEqual(brain.n_vertex, (2, 4))

    def test_unknown_dtype_raises_value_error(self) -> None:
        """Raise on unsupported BrainData dtype values."""
        with self.assertRaises(ValueError):
            fmriprep.BrainData("dummy.nii.gz", dtype="unsupported")

class TestLabelMapperMock(unittest.TestCase):
    """Tests for mock label-mapper behavior."""

    def test_get_value_returns_expected_numeric_mapping(self) -> None:
        """Map a known label to the expected numeric code."""
        mapper = fmriprep.LabelMapper({"stimulus_name": {"face": 1, "scene": 2}})
        self.assertEqual(mapper.get_value("stimulus_name", "face"), 1)
        self.assertTrue(np.isnan(mapper.get_value("stimulus_name", "n/a")))

    def test_get_value_returns_existing_value_for_duplicate_mapping(self) -> None:
        """Allow duplicate labels to resolve to the shared numeric value."""
        mapper = fmriprep.LabelMapper({"stimulus_name": {"face": 1, "scene": 1}})
        self.assertEqual(mapper.get_value("stimulus_name", "face"), 1)
        self.assertEqual(mapper.get_value("stimulus_name", "scene"), 1)

    def test_get_value_raises_on_missing_key(self) -> None:
        """Raise when querying an unmapped metadata key."""
        mapper = fmriprep.LabelMapper({"stimulus_name": {"face": 1}})
        with self.assertRaises(RuntimeError):
            mapper.get_value("condition_name", "face")

class TestCreateBdataFmriprepSubjectMock(MockDatasetMixin):
    """Tests for single-subject mock BData creation."""

    def _num_voxels(self) -> int:
        """Return the voxel count in the mock volume."""
        return int(np.prod(DATA_BUILDER.voxel_shape))

    def _prepare_confounds_with_nan(self, confounds_path: Path) -> None:
        """Inject NaNs and an extra column into a confounds file."""
        conf_df = DATA_BUILDER.confounds_map[str(confounds_path)].copy()
        conf_df["framewise_displacement"] = np.nan
        conf_df["cosine_extra"] = np.arange(len(conf_df))
        conf_df.to_csv(confounds_path, sep="\t", index=False)

    def test_confounds_with_nan_and_extra_columns(self) -> None:
        """Preserve NaNs and tolerate extra confounds columns."""
        confounds_path = self.data_root / DATA_BUILDER.expected_runs["ses-01"][0]["confounds"]
        self._prepare_confounds_with_nan(confounds_path)
        fmriprep_data = fmriprep.FmriprepData(self.data_root.as_posix())
        subject_data = fmriprep_data.data[self.subject]
        brain_data = _get_private("create_bdata_fmriprep_subject")(
            subject_data=subject_data,
            data_path=self.data_root.as_posix(),
            data_mode="volume_native",
            label_mapper={"stimulus_name": {"face": 1, "scene": 2}},
            with_confounds=True,
        )
        confounds = brain_data.get("Confounds")
        self.assertEqual(confounds.shape[0], brain_data.get("VoxelData").shape[0])
        self.assertTrue(np.isnan(confounds).any())

    def test_create_bdata_fmriprep_subject(self) -> None:
        """Create single-subject BData with expected core matrices."""
        fmriprep_data = fmriprep.FmriprepData(self.data_root.as_posix())
        subject_data = fmriprep_data.data[self.subject]
        brain_data = _get_private("create_bdata_fmriprep_subject")(
            subject_data=subject_data,
            data_path=self.data_root.as_posix(),
            data_mode="volume_native",
            label_mapper={"stimulus_name": {"face": 1, "scene": 2}},
            with_confounds=True,
        )
        self.assertEqual(brain_data.get("VoxelData").shape[1], self._num_voxels())
        self.assertEqual(brain_data.get("Session").shape[1], 1)
        self.assertEqual(brain_data.get("Run").shape[1], 1)
        self.assertEqual(brain_data.get("Block").shape[1], 1)
        np.testing.assert_array_equal(brain_data.get("stimulus_name")[:, 0], brain_data.get("Label")[:, 1])

    def test_create_bdata_singlesubject(self) -> None:
        """Create a single-subject BData via the public helper."""
        fmriprep_data = fmriprep.FmriprepData(self.data_root.as_posix())
        subject_data = fmriprep_data.data[self.subject]
        brain_data = fmriprep.create_bdata_singlesubject(
            subject_data=subject_data,
            data_path=self.data_root.as_posix(),
            data_mode="volume_native",
            label_mapper={"stimulus_name": {"face": 1, "scene": 2}},
            with_confounds=True,
        )
        self.assertEqual(brain_data.get("VoxelData").shape[1], self._num_voxels())
        self.assertEqual(brain_data.get("Confounds").shape[0], brain_data.get("VoxelData").shape[0])

class TestCutRunMock(unittest.TestCase):
    """Tests for run truncation behavior with mock event files."""

    def setUp(self) -> None:
        """Create a one-run dataset with an overlong final event."""
        builder = MockBidsBuilder()
        builder.sessions = ("ses-01",)
        builder.run_numbers = (1,)
        builder.build()
        events_path = next(iter(builder.events_map.keys()))
        events_df = builder.events_map[events_path].copy()
        events_df.loc[events_df.index[-1], "duration"] = events_df.loc[events_df.index[-1], "duration"] + 1
        events_df.to_csv(events_path, sep="\t", index=False)
        builder.events_map[events_path] = events_df
        self.builder = builder
        self.data_root = builder.root
        self.subject = builder.subject

    def tearDown(self) -> None:
        """Clean up the temporary mock dataset."""
        self.builder.cleanup()

    def test_cut_run_false_raises(self) -> None:
        """Raise when event timing exceeds volumes and cut_run is disabled."""
        fmriprep_data = fmriprep.FmriprepData(self.data_root.as_posix())
        subject_data = fmriprep_data.data[self.subject]
        with self.assertRaises(ValueError):
            _get_private("create_bdata_fmriprep_subject")(
                subject_data=subject_data,
                data_path=self.data_root.as_posix(),
                data_mode="volume_native",
                label_mapper={"stimulus_name": {"face": 1, "scene": 2}},
                with_confounds=False,
                cut_run=False,
            )

    def test_cut_run_true_truncates(self) -> None:
        """Truncate samples when event timing exceeds available volumes."""
        fmriprep_data = fmriprep.FmriprepData(self.data_root.as_posix())
        subject_data = fmriprep_data.data[self.subject]
        brain_data = _get_private("create_bdata_fmriprep_subject")(
            subject_data=subject_data,
            data_path=self.data_root.as_posix(),
            data_mode="volume_native",
            label_mapper={"stimulus_name": {"face": 1, "scene": 2}},
            with_confounds=False,
            cut_run=True,
        )
        expected_samples = int(next(iter(self.builder.data_map.values())).shape[-1])
        self.assertEqual(brain_data.get("VoxelData").shape[0], expected_samples)
        self.assertEqual(brain_data.get("Block").shape[0], expected_samples)
        self.assertEqual(brain_data.get("Label").shape[0], expected_samples)

class TestGetXyzMock(unittest.TestCase):
    """Tests for voxel-coordinate extraction helpers."""

    def setUp(self) -> None:
        """Patch xrange compatibility for the private helper."""
        self._get_xyz = _get_private("get_xyz")
        self._had_xrange = hasattr(fmriprep, "xrange")
        self._original_xrange = fmriprep.xrange if self._had_xrange else None
        fmriprep.xrange = range

    def tearDown(self) -> None:
        """Restore the original xrange compatibility hook."""
        if not self._had_xrange:
            fmriprep.__dict__.pop("xrange", None)
        else:
            fmriprep.xrange = self._original_xrange

    def test_get_xyz_with_4d_image(self) -> None:
        """Compute XYZ coordinates correctly for 4D-like images."""
        affine_5d = np.eye(5)

        class FakeCoordMap:
            affine = affine_5d

        class FakeImage:
            shape = (2, 2, 1, 2)
            coordmap = FakeCoordMap()

        xyz = self._get_xyz(FakeImage())
        affine = np.delete(np.delete(affine_5d, 3, axis=0), 3, axis=1)
        expected_columns = []
        for i, j, k in itertools.product(range(2), range(2), range(1)):
            vec = np.array([i, j, k, 1.0])
            expected_columns.append(affine @ vec)  # type: ignore[operator]
        expected = np.column_stack(expected_columns)[:3]
        np.testing.assert_allclose(xyz, expected)

    def test_get_xyz_with_3d_image(self) -> None:
        """Compute XYZ coordinates correctly for 3D images."""
        affine = np.array([[1.0, 0.0, 0.0, 5.0], [0.0, 2.0, 0.0, 6.0], [0.0, 0.0, 3.0, 7.0], [0.0, 0.0, 0.0, 1.0]])

        class FakeCoordMap:
            def __init__(self, mat: np.ndarray) -> None:
                self.affine = mat

        class FakeImage:
            shape = (2, 1, 2)

            def __init__(self, mat: np.ndarray) -> None:
                self.coordmap = FakeCoordMap(mat)

        xyz = self._get_xyz(FakeImage(affine))
        expected_columns = []
        for i, j, k in itertools.product(range(2), range(1), range(2)):
            vec = np.array([i, j, k, 1.0])
            expected_columns.append(affine @ vec)  # type: ignore[operator]
        expected = np.column_stack(expected_columns)[:3]
        np.testing.assert_allclose(xyz, expected)

class TestLoadMriMock(unittest.TestCase):
    """Tests for loading MRI volumes from mock files."""

    def setUp(self) -> None:
        """Expose the private MRI loading helper."""
        self._load_mri = _get_private("load_mri")

    def test_load_mri_from_4d_image(self) -> None:
        """Load 4D MRI data into flattened sample-by-voxel form."""
        run = DATA_BUILDER.expected_runs["ses-01"][0]
        path = (DATA_BUILDER.root / run["volume_native"]).as_posix()
        data = DATA_BUILDER.data_map[str(DATA_BUILDER.root / run["volume_native"])]
        loaded_data, xyz, ijk = self._load_mri(path)
        expected_data = data.reshape(-1, data.shape[-1], order="F").T
        np.testing.assert_allclose(loaded_data, expected_data)
        ijk_expected = np.array(np.unravel_index(np.arange(np.prod(data.shape[:3])), data.shape[:3], order="F"))
        affine = np.eye(4)
        xyz_expected = (affine @ np.vstack([ijk_expected, np.ones((1, ijk_expected.shape[1]))]))[:-1]
        np.testing.assert_allclose(xyz, xyz_expected)
        np.testing.assert_array_equal(ijk, ijk_expected)

    def test_load_mri_from_3d_image(self) -> None:
        """Load 3D MRI data and preserve voxel coordinates."""
        affine = np.array(
            [
                [1.0, 0.0, 0.0, 5.0],
                [0.0, 2.0, 0.0, 6.0],
                [0.0, 0.0, 3.0, 7.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        data = np.arange(8, dtype=float).reshape(2, 2, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "image.nii.gz"
            nib.save(nib.Nifti1Image(data, affine), path)
            loaded_data, xyz, ijk = self._load_mri(path.as_posix())
        expected_data = data.flatten(order="F")
        np.testing.assert_allclose(loaded_data, expected_data)
        ijk_expected = np.array(np.unravel_index(np.arange(data.size), data.shape, order="F"))
        xyz_expected = (affine @ np.vstack([ijk_expected, np.ones((1, ijk_expected.shape[1]))]))[:-1]
        np.testing.assert_allclose(xyz, xyz_expected)
        np.testing.assert_array_equal(ijk, ijk_expected)

    def test_load_mri_raises_for_invalid_dimension(self) -> None:
        """Raise when the image dimensionality is unsupported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((2, 2)), np.eye(4)), path)
            with self.assertRaises((ValueError, NiftiError)):
                self._load_mri(path.as_posix())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
