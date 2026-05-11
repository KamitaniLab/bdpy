"""Tests for the fMRIPrep real-data processing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from .test_fmriprep_utils import (
    CREATE_GOLDEN_MASTER,
    REAL_EXPECTED_H5,
    RealDatasetMixin,
    fmriprep,
)


@pytest.mark.real_data
class TestCreateBdataFmriprepReal(RealDatasetMixin):
    """Test the create_bdata_fmriprep function with real fMRIPrep output data."""

    def test_create_bdata_fmriprep_real_exclude_gm(self) -> None:
        """Compare real-data output against the golden master."""
        # Update the golden master when adding or changing real-data coverage.
        exclude = {"session/run": [[1, 2, 3, 4, 5], [1]]}

        bdata_list, data_labels_list = fmriprep.create_bdata_fmriprep(
            dpath=self.data_root.as_posix(),
            data_mode="volume_native",
            fmriprep_dir="derivatives/fmriprep",
            label_mapper=self.label_mapper,
            exclude=exclude,
            split_task_label=True,
            with_confounds=True,
            return_data_labels=True,
            return_list=True,
        )
        self.assertEqual(len(data_labels_list), 1)
        self.assertEqual(len(bdata_list), 1)
        brain_data = bdata_list[0]

        if CREATE_GOLDEN_MASTER:
            REAL_EXPECTED_H5.parent.mkdir(parents=True, exist_ok=True)
            brain_data.save(str(REAL_EXPECTED_H5))
            return None

        for key in self.check_keys:
            actual = np.asarray(brain_data.get(key))
            expected = np.asarray(self.expected_bdata.get(key))
            np.testing.assert_array_equal(actual, expected)
