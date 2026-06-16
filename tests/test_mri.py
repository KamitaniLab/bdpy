"""Tests for bdpy.util."""

import unittest

import numpy as np

import bdpy.mri as bmr


class TestMri(unittest.TestCase):
    """Tests for 'mri' module."""

    def test_get_roiflag_pass0001(self) -> None:
        """Test for get_roiflag (pass case 0001)."""
        roi_xyz = [np.array([[1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3]])]
        epi_xyz = np.array([[1, 2, 3, 4, 5, 6],
                            [1, 2, 3, 4, 5, 6],
                            [1, 2, 3, 4, 5, 6]])

        exp_output = np.array([1, 1, 1, 0, 0, 0])

        test_output = bmr.get_roiflag(roi_xyz, epi_xyz) # type: ignore

        self.assertTrue((test_output == exp_output).all())

    def test_load_mri_3d(self) -> None:
        """Test load_mri on a 3D volume.

        Guards the NumPy 2.0 / nibabel 5.x compatibility fix: nibabel removed
        ``get_data()``, so the loader reads the image via ``get_fdata()``.
        """
        import os
        import tempfile

        import nibabel

        arr = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, 'test.nii.gz')
            nibabel.save(nibabel.Nifti1Image(arr, affine=np.eye(4)), fpath)
            data, xyz, ijk = bmr.load_mri(fpath)  # type: ignore

        # A 3D volume is returned flattened in Fortran order.
        np.testing.assert_array_equal(data, arr.flatten(order='F'))
        self.assertEqual(xyz.shape, (3, arr.size))
        self.assertEqual(ijk.shape, (3, arr.size))

    def test_get_roiflag_pass0002(self) -> None:
        """Test for get_roiflag (pass case 0002)."""
        roi_xyz = [np.array([[1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3]]),
                   np.array([[5, 6],
                             [5, 6],
                             [5, 6]])]
        epi_xyz = np.array([[1, 2, 3, 4, 5, 6],
                            [1, 2, 3, 4, 5, 6],
                            [1, 2, 3, 4, 5, 6]])

        exp_output = np.array([[1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 1]])

        test_output = bmr.get_roiflag(roi_xyz, epi_xyz) # type: ignore

        self.assertTrue((test_output == exp_output).all())


if __name__ == '__main__':
    unittest.main()
