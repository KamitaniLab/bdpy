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

    def test_load_mri_4d(self) -> None:
        """Test load_mri on a 4D volume.

        Guards a bug in the 4D path where the affine was reduced to 3x3 before
        being applied to homogeneous (4 x N) voxel indices, raising a shape
        mismatch. The 4D path must use the full 4x4 affine, just like the 3D
        path, and return one row per time point.
        """
        import os
        import tempfile

        import nibabel

        i_len, j_len, k_len, t_len = 2, 3, 4, 5
        n_vox = i_len * j_len * k_len
        arr = np.arange(n_vox * t_len, dtype=np.float32).reshape(
            i_len, j_len, k_len, t_len)
        affine = np.array([[2., 0., 0., -10.],
                           [0., 2., 0., -12.],
                           [0., 0., 2., -8.],
                           [0., 0., 0., 1.]])

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, 'test4d.nii.gz')
            nibabel.save(nibabel.Nifti1Image(arr, affine=affine), fpath)
            data, xyz, ijk = bmr.load_mri(fpath)  # type: ignore

        # Data is (sample x voxel): one sample per time point, voxels flattened
        # in Fortran order.
        self.assertEqual(data.shape, (t_len, n_vox))
        np.testing.assert_array_equal(
            data, arr.reshape(-1, t_len, order='F').T)

        # ijk are the voxel indices in Fortran order.
        expected_ijk = np.array(np.unravel_index(
            np.arange(n_vox), (i_len, j_len, k_len), order='F'))
        np.testing.assert_array_equal(ijk, expected_ijk)

        # xyz are the world coordinates from the full 4x4 affine.
        ijk_b = np.vstack([expected_ijk, np.ones((1, n_vox))])
        expected_xyz = np.dot(affine, ijk_b)[:3]
        np.testing.assert_allclose(xyz, expected_xyz)
        self.assertEqual(xyz.shape, (3, n_vox))

    def test_braindata_load_volume_4d(self) -> None:
        """Test BrainData volume loading on a 4D volume.

        Guards the same 4D affine bug as test_load_mri_4d in the duplicated
        loader inside fmriprep.BrainData.__load_volume. The surrounding
        create_bdata_fmriprep code treats ``data.shape[0]`` as the number of
        volumes (time points), so the 4D path must return ``(T, N)`` data and
        spatial xyz from the full 4x4 affine.
        """
        import os
        import tempfile

        import nibabel

        from bdpy.mri.fmriprep import BrainData

        i_len, j_len, k_len, t_len = 2, 3, 4, 5
        n_vox = i_len * j_len * k_len
        arr = np.arange(n_vox * t_len, dtype=np.float32).reshape(
            i_len, j_len, k_len, t_len)
        affine = np.array([[2., 0., 0., -10.],
                           [0., 2., 0., -12.],
                           [0., 0., 2., -8.],
                           [0., 0., 0., 1.]])

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, 'bold4d.nii.gz')
            nibabel.save(nibabel.Nifti1Image(arr, affine=affine), fpath)
            brain = BrainData(fpath, dtype='volume')

        # data.shape[0] must be the number of time points (volumes).
        self.assertEqual(brain.data.shape, (t_len, n_vox))
        np.testing.assert_array_equal(
            brain.data, arr.reshape(-1, t_len, order='F').T)

        expected_ijk = np.array(np.unravel_index(
            np.arange(n_vox), (i_len, j_len, k_len), order='F'))
        np.testing.assert_array_equal(brain.index, expected_ijk)

        ijk_b = np.vstack([expected_ijk, np.ones((1, n_vox))])
        expected_xyz = np.dot(affine, ijk_b)[:3]
        np.testing.assert_allclose(brain.xyz, expected_xyz)
        self.assertEqual(brain.xyz.shape, (3, n_vox))

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
