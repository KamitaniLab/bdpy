"""
Utilities for ROIs
"""

import os
import hashlib

import numpy as np
import nibabel.freesurfer

from bdpy.mri import load_mri


def add_roimask(bdata, roi_mask, roi_prefix='',
                brain_data='VoxelData', xyz=['voxel_x', 'voxel_y', 'voxel_z'],
                return_roi_flag=False,
                verbose=True):
    '''Add an ROI mask to `bdata`.

    Parameters
    ----------
    bdata : BData
    roi_mask : str or list
        ROI mask file(s).

    Returns
    -------
    bdata : BData
    '''

    if isinstance(roi_mask, str):
        roi_mask = [roi_mask]

    # Get voxel xyz coordinates in `bdata`
    voxel_xyz = np.vstack([bdata.get_metadata(xyz[0], where=brain_data),
                           bdata.get_metadata(xyz[1], where=brain_data),
                           bdata.get_metadata(xyz[2], where=brain_data)])

    # Load the ROI mask files
    mask_xyz_all = []
    mask_v_all = []

    voxel_consistency = True

    for m in roi_mask:
        mask_v, mask_xyz, mask_ijk = load_mri(m)
        mask_v_all.append(mask_v)
        mask_xyz_all.append(mask_xyz[:, (mask_v == 1).flatten()])

        if voxel_xyz.shape != mask_xyz.shape or not (voxel_xyz == mask_xyz).all():
            voxel_consistency = False

    # Get ROI flags
    if voxel_consistency:
        roi_flag = np.vstack(mask_v_all)
    else:
        roi_flag = get_roiflag(mask_xyz_all, voxel_xyz)

    # Add the ROI flag as metadata in `bdata`
    for i, roi in enumerate(roi_mask):
        roi_name = roi_prefix + '_' + os.path.basename(roi).replace('.nii.gz', '').replace('.nii', '')

        with open(roi, 'rb') as f:
            roi_md5 = hashlib.md5(f.read()).hexdigest()

        roi_desc = '1 = ROI %s (source file: %s; md5: %s)' % (roi_name, roi, roi_md5)

        print('Adding %s' % roi_name)
        print('  %s' % roi_desc)
        bdata.add_metadata(roi_name, roi_flag[i, :], description=roi_desc, where=brain_data)

    if return_roi_flag:
        return bdata, roi_flag
    else:
        return bdata


def get_roiflag(roi_xyz_list, epi_xyz_array, verbose=True):
    """
    Get ROI flags

    Parameters
    ----------
    roi_xyz_list : list, len = n_rois
        List of arrays that contain XYZ coordinates of ROIs. Each element is an
        array of shape = (3, n_voxels_in_roi).
    epi_xyz_array : array, shape = (3, n_voxels)
        Voxel XYZ coordinates
    verbose : boolean
        If True, 'get_roiflag' outputs verbose message

    Returns
    -------
    roi_flag : array, shape = (n_rois, n_voxels)
        ROI flag array
    """

    epi_voxel_size = epi_xyz_array.shape[1]

    if verbose:
        print("EPI num voxels: %d" % epi_voxel_size)

    roi_flag_array = np.zeros((len(roi_xyz_list), epi_voxel_size))

    epi_xyz_dist = np.sum(epi_xyz_array ** 2, axis=0)

    for i, roi_xyz in enumerate(roi_xyz_list):
        if verbose:
            print("ROI %d num voxels: %d" % (i + 1, len(roi_xyz[0])))

        roi_xyz_dist = np.sum(roi_xyz ** 2, axis=0)

        roi_flag_temp = np.zeros(epi_xyz_dist.shape)

        for j, mv_dist in enumerate(roi_xyz_dist):
            candidate_index = epi_xyz_dist == mv_dist
            roi_flag_in_candidate = [np.array_equal(v_xyz, roi_xyz[:, j].flatten())
                                     for v_xyz in epi_xyz_array[:, candidate_index].T]
            roi_flag_temp[candidate_index] = roi_flag_temp[candidate_index] + roi_flag_in_candidate

        roi_flag_temp[roi_flag_temp > 1] = 1
        roi_flag_array[i, :] = roi_flag_temp

    return roi_flag_array


def add_roilabel(bdata, label, vertex_data=['VertexData'], prefix='', verbose=False):
    '''Add ROI label(s) to `bdata`.

    Parameters
    ----------
    bdata : BData
    roi_mask : str or list
        ROI label file(s).

    Returns
    -------
    bdata : BData
    '''

    def add_roilabel_file(bdata, label, vertex_data=['VertexData'], prefix='', verbose=False):
        # Read the label file
        roi_vertex = nibabel.freesurfer.read_label(label)

        # Make meta-data vector for ROI flag
        vindex = bdata.get_metadata('vertex_index', where=vertex_data)

        roi_flag = np.zeros(vindex.shape)

        for v in roi_vertex:
            roi_flag[vindex == v] = 1

        roi_name = prefix + '_' + os.path.basename(label).replace('.label', '')

        with open(label, 'rb') as f:
            roi_md5 = hashlib.md5(f.read()).hexdigest()

        roi_desc = '1 = ROI %s (source file: %s; md5: %s)' % (roi_name, label, roi_md5)
        if verbose:
            print('Adding %s (%d vertices)' % (roi_name, np.sum(roi_flag)))
            print('  %s' % roi_desc)
        bdata.add_metadata(roi_name, roi_flag, description=roi_desc, where=vertex_data)

        return bdata

    if isinstance(label, str):
        label = [label]

    for lb in label:
        if os.path.splitext(lb)[1] == '.label':
            # FreeSurfer label file
            bdata = add_roilabel_file(bdata, lb, vertex_data=vertex_data, prefix=prefix, verbose=verbose)
        elif os.path.splitext(lb)[1] == '.annot':
            # FreeSurfer annotation file
            annot = nibabel.freesurfer.read_annot(lb)
            labels = annot[0]  # Annotation ID at each vertex (shape = (n_vertices,))
            ctab = annot[1]    # Label color table (RGBT + label ID)
            names = annot[2]   # Label name list

            with open(lb, 'rb') as f:
                roi_md5 = hashlib.md5(f.read()).hexdigest()

            for i, name in enumerate(names):
                label_id = i  # Label ID is zero-based
                roi_flag = (labels == label_id).astype(int)

                if sum(roi_flag) == 0:
                    if verbose:
                        print('Label %s not found in the surface.' % name)
                    continue

                # FIXME: better way to decide left/right?
                if 'Left' in vertex_data:
                    hemi = 'lh'
                elif 'Right' in vertex_data:
                    hemi = 'rh'
                else:
                    raise ValueError('Invalid vertex_data: %s' % vertex_data)

                roi_name = prefix + '_' + hemi + '.' + name
                roi_desc = '1 = ROI %s (source file: %s; md5: %s)' % (roi_name, lb, roi_md5)

                if verbose:
                    print('Adding %s (%d vertices)' % (roi_name, np.sum(roi_flag)))
                    print('  %s' % roi_desc)

                bdata.add_metadata(roi_name, roi_flag, description=roi_desc, where=vertex_data)
        else:
            raise TypeError('Unknown file type: %s' % os.path.splitext(lb)[0])

    return bdata
