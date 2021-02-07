"""
Utilities for ROIs
"""

import os
import glob
import re
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
    md_keys = []
    md_descs = []

    for i, roi in enumerate(roi_mask):
        roi_name = roi_prefix + '_' + os.path.basename(roi).replace('.nii.gz', '').replace('.nii', '')

        with open(roi, 'rb') as f:
            roi_md5 = hashlib.md5(f.read()).hexdigest()

        roi_desc = '1 = ROI %s (source file: %s; md5: %s)' % (roi_name, roi, roi_md5)

        print('Adding %s' % roi_name)
        print('  %s' % roi_desc)
        md_keys.append(roi_name)
        md_descs.append(roi_desc)

    bdata.metadata.key.extend(md_keys)
    bdata.metadata.description.extend(md_descs)

    brain_data_index = bdata.get_metadata(brain_data)
    new_md_v = np.zeros([roi_flag.shape[0], bdata.metadata.value.shape[1]])
    new_md_v[:, :] = np.nan
    new_md_v[:, brain_data_index == 1] = roi_flag

    bdata.metadata.value = np.vstack([bdata.metadata.value, new_md_v])

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


def add_rois(bdata, roi_files, data_type='volume', prefix_map={}):
    '''Add ROIs in bdata from files.'''

    roi_prefix_from_annot = {'lh.aparc.a2009s.annot': 'freesurfer_destrieux',
                             'rh.aparc.a2009s.annot': 'freesurfer_destrieux',
                             'lh.aparc.annot': 'freesurfer_dk',
                             'rh.aparc.annot': 'freesurfer_dk'}

    if data_type == 'volume':
        # List all ROI mask files
        roi_files_all = []
        for roi_files_pt in roi_files:
            roi_files_all.extend(glob.glob(roi_files_pt))

        # Group ROI mask files by directories (prefix)
        roi_group = {}
        for roi_file in roi_files_all:
            roi_prefix = os.path.basename(os.path.dirname(roi_file))
            if roi_prefix in roi_group.keys():
                roi_group[roi_prefix].append(roi_file)
            else:
                roi_group.update({roi_prefix: [roi_file]})

        # Add ROIs
        roi_flag_all = []
        for roi_prefix, roi_files_group in roi_group.items():
            print('Adding ROIs %s' % roi_prefix)
            bdata, roi_flag= add_roimask(bdata, roi_files_group, roi_prefix=roi_prefix, return_roi_flag=True, verbose=True)
            roi_flag_all.append(roi_flag)
        print('')

        # Remove voxels out of ROIs
        roi_flag_all = np.vstack(roi_flag_all)
        remove_voxel_ind = np.sum(roi_flag_all, axis=0) == 0
        _, voxel_ind = bdata.select('VoxelData = 1', return_index=True)
        remove_column_ind = np.where(voxel_ind)[0][remove_voxel_ind]

        bdata.dataset = np.delete(bdata.dataset, remove_column_ind, 1)
        bdata.metadata.value = np.delete(bdata.metadata.value, remove_column_ind, 1)
        # FIXME: needs cleaning

    elif data_type == 'surface':
        # List all ROI labels files
        roi_files_lh_all = []
        roi_files_rh_all = []
        for roi_files_pt in roi_files:
            roi_files_lh_all.extend(glob.glob(roi_files_pt[0]))
            roi_files_rh_all.extend(glob.glob(roi_files_pt[1]))

        for roi_file_lh, roi_file_rh in zip(roi_files_lh_all, roi_files_rh_all):
            _, ext = os.path.splitext(roi_file_lh)
            if ext == '.annot':
                roi_prefix_lh = re.sub('^lh\.', '', os.path.splitext(os.path.basename(roi_file_lh))[0])
            else:
                roi_prefix_lh = os.path.basename(os.path.dirname(roi_file_lh))
            _, ext = os.path.splitext(roi_file_rh)
            if ext == '.annot':
                roi_prefix_rh = re.sub('^rh\.', '', os.path.splitext(os.path.basename(roi_file_rh))[0])
            else:
                roi_prefix_rh = os.path.basename(os.path.dirname(roi_file_rh))

            if roi_prefix_lh != roi_prefix_rh:
                raise ValueError('The left and right hemi labels should share the same prefix.')

            if roi_prefix_lh in prefix_map.keys():
                roi_prefix_lh = prefix_map[roi_prefix_lh]
                roi_prefix_rh = prefix_map[roi_prefix_rh]

            print('Adding ROIs %s' % roi_prefix_lh)
            bdata = add_roilabel(bdata, roi_file_lh, vertex_data='VertexLeft', prefix=roi_prefix_lh, verbose=True)
            bdata = add_roilabel(bdata, roi_file_rh, vertex_data='VertexRight', prefix=roi_prefix_rh, verbose=True)

    else:
        raise ValueError('Invalid data type: %s' % data_type)

    return bdata


def merge_rois(bdata, roi_name, merge_expr):
    '''Merage ROIs.'''

    print('Adding merged ROI %s' % roi_name)

    # Get tokens
    tokens_raw = merge_expr.split(' ')
    tokens_raw = [t for t in tokens_raw if t != '']

    tokens = []
    for tkn in tokens_raw:
        if tkn == '+' or tkn == '-':
            tokens.append(tkn)
        else:
            tkn_e = re.escape(tkn)
            tkn_e = tkn_e.replace('\*', '.*')

            mks = [k for k in bdata.metadata.key if re.match(tkn_e, k)]
            if len(mks) == 0:
                raise RuntimeError('ROI %s not found' % merge_expr)
            s = ' + '.join(mks)
            tokens.extend(s.split(' '))

    print('Merging ROIs: ' + ' '.join(tokens))

    # Parse tokens
    op_stack = []
    rpn_que = []
    for tkn in tokens:
        if tkn == '+' or tkn == '-':
            while op_stack:
                rpn_que.append(op_stack.pop())
            op_stack.append(tkn)
        else:
            rpn_que.append(tkn)
    while op_stack:
        rpn_que.append(op_stack.pop())

    # Get merged ROI meta-data
    out_stack = []
    for tkn in rpn_que:
        if tkn == '+':
            b = out_stack.pop()
            a = out_stack.pop()
            out = a + b
        elif tkn == '-':
            b = out_stack.pop()
            a = out_stack.pop()
            out = a - b
        else:
            out = bdata.get_metadata(tkn)
        out[np.isnan(out)] = 0
        out[out > 1] = 1
        out[out < 0] = 0
        out_stack.append(out)

    if len(out_stack) != 1:
        raise RuntimeError('Something goes wrong in merge_rois.')

    merged_roi_mv = out_stack[0]
    description = 'Merged ROI: %s' % ' '.join(tokens)
    bdata.add_metadata(roi_name, merged_roi_mv, description)

    num_voxels = np.nansum(merged_roi_mv).astype(int)
    print('Num voxels or vertexes: %d' % num_voxels)

    return bdata
