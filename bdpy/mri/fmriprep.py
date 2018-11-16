'''Utilities for fmriprep.'''


import csv
import glob
import itertools
import os
import re
import json
from collections import OrderedDict

import numpy as np
import nipy
import nibabel
import pandas as pd

import bdpy


class FmriprepData(object):
    '''FMRIPREP data class.'''

    def __init__(self, datapath=None, fmriprep_version='1.2'):
        self.__datapath = datapath
        self.__data = OrderedDict()
        self.__fmriprep_version = fmriprep_version

        if self.__datapath is not None:
            self.__parse_data()
            self.__get_task_event_files()

        return None

    @property
    def data(self):
        return self.__data

    # Private methods --------------------------------------------------------

    def __parse_data(self):
        # FMRIPREP results directory
        prepdir = os.path.join(self.__datapath, 'derivatives', 'fmriprep', 'fmriprep')

        # Get subjects
        subjects = self.__get_subjects(prepdir)

        # Get sessions
        for sbj in subjects:
            self.__data.update({sbj: OrderedDict()})
            sessions = self.__get_sessions(prepdir, sbj)

            # Get runs in the sesssion
            for ses in sessions:
                if self.__fmriprep_version == '1.2':
                    if ses == 'ses-anat':
                        continue
                runs = self.__parse_session(prepdir, sbj, ses)
                self.__data[sbj].update({ses: runs})

        return None

    def __get_subjects(self, dpath):
        subjects = []
        for d in os.listdir(dpath):
            if not os.path.isdir(os.path.join(dpath, d)):
                continue

            m = re.match('^sub-.*', d)
            if not m:
                continue

            subjects.append(d)
        return subjects

    def __get_sessions(self, dpath, subject):
        subpath = os.path.join(dpath, subject)
        sessions = []
        for d in os.listdir(subpath):
            if not os.path.isdir(os.path.join(subpath, d)):
                continue

            m = re.match('^ses-.*', d)
            if not m:
                continue

            sessions.append(d)
        return sorted(sessions)

    def __parse_session(self, dpath, subject, session):
        sespath = os.path.join(dpath, subject, session)
        funcpath = os.path.join(sespath, 'func')

        # File name patterns
        # FIXME
        if self.__fmriprep_version == '1.2':
            file_pattern = {'volume_native'   : '.*_space-T1w_desc-preproc_bold\.nii\.gz$',
                            'volume_standard' : '.*_space-MNI152NLin2009cAsym_desc-preproc_bold\.nii\.gz$',
                            'surf_native_left'  : '.*_space-fsnative_hemi-L\.func\.gii$',
                            'surf_native_right' : '.*_space-fsnative_hemi-R\.func\.gii$',
                            'surf_standard_left'  : '.*_space-fsaverage_hemi-L\.func\.gii$',
                            'surf_standard_right' : '.*_space-fsaverage_hemi-R\.func\.gii$',
                            'confounds'       : '.*_desc-confounds_regressors\.tsv$'}
        elif self.__fmriprep_version in ['1.0', '1.1']:
            file_pattern = {'volume_native'   : '.*_bold_space-T1w_preproc\.nii\.gz$',
                            'volume_standard' : '.*_bold_space-MNI152NLin2009cAsym_preproc\.nii\.gz$',
                            'surf_native_left'  : '.*_space-fsnative\.L\.func\.gii$',
                            'surf_native_right' : '.*_space-fsnative\.R\.func\.gii$',
                            'surf_standard_left'  : '.*_space-fsaverage\.L\.func\.gii$',
                            'surf_standard_right' : '.*_space-fsaverage\.R\.func\.gii$',
                            'confounds'       : '.*_bold_confounds\.tsv'}
        else:
            raise ValueError('Unsuppored fmriprep version %s' % self.__fmriprep_version)

        prep_vol_native = []    # List of preprocessed EPI files in native space (T1w)
        prep_vol_standard = []  # List of preprocessed EPI files in MNI space

        prep_surf_native_left = []     # List of preprocessed EPI files (surf) on native surface left
        prep_surf_native_right = []    # List of preprocessed EPI files (surf) on native surface right
        prep_surf_standard_left = []   # List of preprocessed EPI files (surf) on fsaverage left
        prep_surf_standard_right = []  # List of preprocessed EPI files (surf) on fsaverage right

        confounds = []          # List of confound files

        # FIXME: TOO DIRTY
        for f in os.listdir(funcpath):
            if os.path.isdir(os.path.join(funcpath, f)):
                continue

            # Get Motion-corrected EPI files in native T1w
            m = re.search(file_pattern['volume_native'], f)
            if m:
                prep_vol_native.append(f)
                continue

            # Get Motion-corrected EPI files in MNI space
            m = re.search(file_pattern['volume_standard'], f)
            if m:
                prep_vol_standard.append(f)
                continue

            # Surface native
            m = re.search(file_pattern['surf_native_left'], f)
            if m:
                prep_surf_native_left.append(f)
                continue

            m = re.search(file_pattern['surf_native_right'], f)
            if m:
                prep_surf_native_right.append(f)
                continue

            # Surface standard
            m = re.search(file_pattern['surf_standard_left'], f)
            if m:
                prep_surf_standard_left.append(f)
                continue

            m = re.search(file_pattern['surf_standard_right'], f)
            if m:
                prep_surf_standard_right.append(f)
                continue

            # Get confound file (*_)
            m = re.search(file_pattern['confounds'], f)
            if m:
                confounds.append(f)
                continue

        prep_vol_native.sort()
        prep_vol_standard.sort()
        prep_surf_native_left.sort()
        prep_surf_native_right.sort()
        prep_surf_standard_left.sort()
        prep_surf_standard_right.sort()
        confounds.sort()

        # TODO: add run num check

        runs = [{'volume_native' : os.path.join('derivatives', 'fmriprep', 'fmriprep', subject, session, 'func', nat),
                 'volume_standard' : os.path.join('derivatives', 'fmriprep', 'fmriprep', subject, session, 'func', std),
                 'surface_native' : (os.path.join('derivatives', 'fmriprep', 'fmriprep', subject, session, 'func', surf_nat_l),
                                     os.path.join('derivatives', 'fmriprep', 'fmriprep', subject, session, 'func', surf_nat_r)),
                 'surface_standard' : (os.path.join('derivatives', 'fmriprep', 'fmriprep', subject, session, 'func', surf_std_l),
                                       os.path.join('derivatives', 'fmriprep', 'fmriprep', subject, session, 'func', surf_std_r)),
                 'confounds' : os.path.join('derivatives', 'fmriprep', 'fmriprep', subject, session, 'func', conf)}
                for nat, std, surf_nat_l, surf_nat_r, surf_std_l, surf_std_r, conf in zip(prep_vol_native, prep_vol_standard, prep_surf_native_left, prep_surf_native_right, prep_surf_standard_left, prep_surf_standard_right, confounds)]

        return runs

    def __get_task_event_files(self):
        for sbj, sbjdata in self.__data.items():
            for ses, sesdata in sbjdata.items():
                raw_func_dir = os.path.join(self.__datapath, sbj, ses, 'func')
                for run in sesdata:
                    # Get run label
                    if self.__fmriprep_version == '1.2':
                        m = re.search('.*_(run-.*)_desc-confounds_.*', run['confounds'])
                    elif self.__fmriprep_version in ['1.0', '1.1']:
                        m = re.search('.*_(run-.*)_bold_.*', run['confounds'])
                    else:
                        raise ValueError('Unsuppored fmriprep version %s' % self.__fmriprep_version)
                    if m:
                        run_label = m.group(1)
                    else:
                        raise RuntimeError('Run not found!')
                    # Get task event file
                    event_file_name_glob = '%s_%s_task-*_%s_events.tsv' % (sbj, ses, run_label)
                    event_file_list = glob.glob(os.path.join(raw_func_dir, event_file_name_glob))
                    if len(event_file_list) != 1:
                        raise RuntimeError('Something is wrong on task event files.')
                    event_file = event_file_list[0].replace(os.path.normpath(self.__datapath) + '/', '')
                    # Add the task event file in data
                    run.update({'task_event_file': event_file})

                    # Get bold json file
                    bold_json_file_name_glob = '%s_%s_task-*_%s_bold.json' % (sbj, ses, run_label)
                    bold_json_file_list = glob.glob(os.path.join(raw_func_dir, bold_json_file_name_glob))
                    if len(bold_json_file_list) != 1:
                        raise RuntimeError('Something is wrong on bold parameter json files.')
                    bold_json_file = bold_json_file_list[0].replace(os.path.normpath(self.__datapath) + '/', '')
                    run.update({'bold_json': bold_json_file})
        return None


def create_bdata_fmriprep(dpath, data_mode='volume_native', fmriprep_version='1.2', label_mapper=None):
    '''Create BData from FMRIPREP outputs.

    Parameters
    ----------
    dpath : str
        Path to a BIDS data directory.
    data_mode : {'volume_standard', 'volume_native', 'surface_standard', 'surface_native'}
        Data to be loaded.
    fmriprep_version : {'1.2, '1.1', '1.0'}
        The version of fmriprep (default: '1.2')
    label_mapper : dict
        A dictionary of tables that define mapping between non-numerical value
        (e.g., string) in task event files and float value in BData.dataset.

    Returns
    -------
    BData or list of BData
        One subject, one BData.
    '''

    print('BIDS data path: %s' % dpath)

    # Label mapper
    if label_mapper is None:
        label_mapper_dict = {}
    else:
        if not isinstance(label_mapper, dict):
            raise TypeError('Unsupported label mapper (type: %s)' % type(label_mapper))

        label_mapper_dict = {}
        for lbmp in label_mapper:
            if not isinstance(label_mapper[lbmp], str):
                raise TypeError('Unsupported label mapper (type: %s)' % type(label_mapper[lbmp]))

            lbmp_file = label_mapper[lbmp]

            ext = os.path.splitext(lbmp_file)[1]
            lbmp_dict = {}

            if ext == '.csv':
                with open(lbmp_file, 'r') as f:
                    reader = csv.reader(f, delimiter=',')
                    for row in reader:
                        lbmp_dict.update({row[0]: int(row[1])})
            elif ext == '.tsv':
                with open(lbmp_file, 'r') as f:
                    reader = csv.reader(f, delimiter='\t')
                    for row in reader:
                        lbmp_dict.update({row[0]: int(row[1])})
            else:
                raise ValueError('Unsuppored label mapper file: %s' % lbmp_file)

            lbmp_dict.update({'n/a': np.nan})
            label_mapper_dict.update({lbmp: lbmp_dict})

    # Create BData from fmriprep outputs
    fmriprep = FmriprepData(dpath, fmriprep_version=fmriprep_version)

    bdata_list = []

    for sbj, sbjdata in fmriprep.data.items():
        print('----------------------------------------')
        print('Subject: %s\n' % sbj)

        bdata = __create_bdata_fmriprep_subject(sbjdata, data_mode, data_path=dpath, label_mapper=label_mapper_dict)
        bdata_list.append(bdata)

    if len(bdata_list) == 1:
        return bdata_list[0]
    else:
        return bdata_list


class BrainData(object):
    '''fMRI data class (volume or surface).'''

    def __init__(self, dpath, dtype='volume'):
        self.__dpath = dpath
        self.__dtype = dtype
        self.__data = np.array([])
        self.__xyz = np.array([])
        self.__index = np.array([])

        self.__n_vertex = (-1, -1)

        if self.__dtype == 'volume':
            self.__load_volume()
        elif self.__dtype == 'surface':
            self.__load_surface()
        else:
            raise ValueError('Unknown dtype: %s' % self.__dtype)

    @property
    def data(self):
        return self.__data

    @property
    def xyz(self):
        if self.__dtype is 'surface':
            raise NotImplementedError('Vertex xyz coordinates are not implemented yet.')
        return self.__xyz

    @property
    def index(self):
        return self.__index

    @property
    def n_vertex(self):
        if self.__dtype is not 'surface':
            raise TypeError('Not surface data.')
        return self.__n_vertex

    def __load_volume(self):
        '''Load a MRI image.

        - Returns data as 2D array (sample x voxel)
        - Returns voxle xyz coordinates (3 x voxel)
        - Returns voxel ijk indexes (3 x voxel)
        - Data, xyz, and ijk are flattened by Fortran-like index order
        '''
        img = nipy.load_image(self.__dpath)

        data = img.get_data()
        if data.ndim == 4:
            data = data.reshape(-1, data.shape[-1], order='F').T
            i_len, j_len, k_len, t = img.shape
            affine = np.delete(np.delete(img.coordmap.affine, 3, axis=0), 3, axis=1)
        elif data.ndim == 3:
            data = data.flatten(order='F')
            i_len, j_len, k_len = img.shape
            affine = img.coordmap.affine
        else:
            raise ValueError('Invalid shape.')

        ijk = np.array(np.unravel_index(np.arange(i_len * j_len * k_len),
                                        (i_len, j_len, k_len), order='F'))
        ijk_b = np.vstack([ijk, np.ones((1, i_len * j_len * k_len))])
        xyz_b = np.dot(affine, ijk_b)
        xyz = xyz_b[:-1]

        self.__data = data
        self.__xyz = xyz
        self.__index = ijk

        return None

    def __load_surface(self):
        print('Loading %s ...' % self.__dpath[0])
        vertex_left = self.__load_surf_func_file(self.__dpath[0])
        print('Loading %s ...' % self.__dpath[1])
        vertex_right = self.__load_surf_func_file(self.__dpath[1])

        n_vertex_left = vertex_left.shape[1]
        n_vertex_right = vertex_right.shape[1]

        # TOOD: check size

        self.__data = np.hstack([vertex_left, vertex_right])
        self.__index = np.array([np.hstack([np.arange(vertex_left.shape[1]),
                                            np.arange(vertex_right.shape[1])])])
        self.__n_vertex = (n_vertex_left, n_vertex_right)

        # TODO: add vertex xyz

        return None

    def __load_surf_func_file(self, fpath):
        surf = nibabel.load(fpath)
        data_arrays = surf.darrays
        data_matrix_list = []
        for d in data_arrays:
            # TODO: checks vertex num
            data_matrix_list.append(d.data)
        data_matrix = np.vstack(data_matrix_list)
        return data_matrix

def __create_bdata_fmriprep_subject(subject_data, data_mode, data_path='./', label_mapper={}):
    if data_mode in ['surface_standard', 'surface_native']:
        is_surf = True
    else:
        is_surf = False

    braindata_list = []
    xyz = np.array([])
    ijk = np.array([])

    motionparam_list = []

    ses_label_list = []
    run_label_list = []
    block_label_list = []
    labels_list = []

    last_run = 0
    last_block = 0

    for i, (ses, sesdata) in enumerate(subject_data.items()):
        print('Session: %d (%s)' % (i + 1, ses))
        print('Data: %s\n' % data_mode)

        for j, run in enumerate(sesdata):
            print('Run %d' % (j + 1))
            epi = run[data_mode]
            event_file = run['task_event_file']
            confounds_file = run['confounds']
            if is_surf:
                print('EPI:             %s, %s' % epi)
            else:
                print('EPI:             %s' % epi)
            print('Task event file: %s' % event_file)
            print('Confounds file:  %s' % confounds_file)

            mp_label_col = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ',
                            'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

            # Load brain data (volume or surface)
            if is_surf:
                brain = BrainData((os.path.join(data_path, epi[0]), os.path.join(data_path, epi[1])), dtype='surface')
            else:
                brain = BrainData(os.path.join(data_path, epi), dtype='volume')
                xyz = brain.xyz

            braindata_list.append(brain.data)
            ijk = brain.index

            num_vol = brain.data.shape[0]

            # Load motion parameters (and the other confounds)
            conf_pd = pd.read_csv(os.path.join(data_path, confounds_file), delimiter='\t')

            mp_label = [c for c in conf_pd.columns if c in mp_label_col]
            if len(mp_label) != 6:
                raise RuntimeError('Invalid confounds file: %s' % os.path.join(data_path, confounds_file))

            mp = np.hstack([np.c_[conf_pd[a]] for a in mp_label])
            motionparam_list.append(mp)

            # Load task event file
            event_file = os.path.join(data_path, run['task_event_file'])
            events = pd.read_csv(event_file, delimiter='\t')

            # Check time length
            tlen_event = events.tail(1)['onset'].values[0] + events.tail(1)['duration'].values[0]
            n_sample = brain.data.shape[0]

            with open(os.path.join(data_path, run['bold_json']), 'r') as f:
                bold_metainfo = json.load(f)
            tr = bold_metainfo['RepetitionTime']

            if tlen_event != n_sample * tr:
                raise ValueError('The number of volumes in the EPI file (%s) '
                                 'and time duration in the corresponding task '
                                 'event file mismatch!'
                                 % (epi, run['task_event_file']))

            # Make block and labels
            blocks = []
            labels = []

            cols = events.columns.values
            cols = cols[~(cols == 'onset')]
            cols = cols[~(cols == 'duration')]

            for k, row in events.iterrows():
                onset = row['onset']
                duration = row['duration']
                nsmp = int(duration / tr) # TODO: fix for float

                # Block
                blocks.append(np.ones((nsmp, 1)) * (k + 1))

                # Label
                label_vals = []
                for p in cols:
                    if p in label_mapper:
                        label_vals.append(label_mapper[p][row[p]])
                    else:
                        label_vals.append(row[p])
                label_vals = np.array([np.nan if x == 'n/a' else np.float(x)
                                       for x in label_vals])
                label_mat = np.tile(label_vals, (nsmp, 1))
                labels.append(label_mat)

            ses_label_list.append(np.ones((num_vol, 1)) * (i + 1))
            run_label_list.append(np.ones((num_vol, 1)) * (j + 1) + last_run)
            block_label_list.append(np.vstack(blocks) + last_block)
            labels_list.append(np.vstack(labels))

            last_block = block_label_list[-1][-1]

        last_run = run_label_list[-1][-1]

        print('')

    braindata = np.vstack(braindata_list)
    motionparam = np.vstack(motionparam_list)
    ses_label = np.vstack(ses_label_list)
    run_label = np.vstack(run_label_list)
    block_label = np.vstack(block_label_list)
    labels_label = np.vstack(labels_list)

    # Create BData (one subject, one file)
    bdata = bdpy.BData()

    if is_surf:
        bdata.add(braindata, 'VertexData')
        n_vertex = brain.n_vertex
        bdata.add_metadata('VertexLeft', np.array([1] * n_vertex[0] + [0] * n_vertex[1]),
                           where='VertexData')
        bdata.add_metadata('VertexRight', np.array([0] * n_vertex[0] + [1] * n_vertex[1]),
                           where='VertexData')
    else:
        bdata.add(braindata, 'VoxelData')

    bdata.add(ses_label, 'Session')
    bdata.add(run_label, 'Run')
    bdata.add(block_label, 'Block')
    bdata.add(labels_label, 'Label')
    bdata.add(motionparam, 'MotionParameter')
    bdata.add_metadata('MotionParameter_trans_x', [1, 0, 0, 0, 0, 0], 'Motion parameter: x translation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_trans_y', [0, 1, 0, 0, 0, 0], 'Motion parameter: y translation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_trans_z', [0, 0, 1, 0, 0, 0], 'Motion parameter: z translation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_rot_x', [0, 0, 0, 1, 0, 0], 'Motion parameter: x rotation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_rot_y', [0, 0, 0, 0, 1, 0], 'Motion parameter: y rotation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_rot_z', [0, 0, 0, 0, 0, 1], 'Motion parameter: z rotation', where='MotionParameter')

    for i, col in enumerate(cols):
        metadata_vec = np.empty((len(cols),))
        metadata_vec[:] = np.nan
        metadata_vec[i] = 1
        bdata.add_metadata(col, metadata_vec, 'Label %s' % col, where='Label')

    if is_surf:
        # bdata.add_metadata('vertex_x', xyz[0, :], 'Vertex x coordinate', where='VertexData')
        # bdata.add_metadata('vertex_y', xyz[1, :], 'Vertex y coordinate', where='VertexData')
        # bdata.add_metadata('vertex_z', xyz[2, :], 'Vertex z coordinate', where='VertexData')
        bdata.add_metadata('vertex_index', ijk[0, :], 'Vertex index', where='VertexData')
    else:
        bdata.add_metadata('voxel_x', xyz[0, :], 'Voxel x coordinate', where='VoxelData')
        bdata.add_metadata('voxel_y', xyz[1, :], 'Voxel y coordinate', where='VoxelData')
        bdata.add_metadata('voxel_z', xyz[2, :], 'Voxel z coordinate', where='VoxelData')
        bdata.add_metadata('voxel_i', ijk[0, :], 'Voxel i index', where='VoxelData')
        bdata.add_metadata('voxel_j', ijk[1, :], 'Voxel j index', where='VoxelData')
        bdata.add_metadata('voxel_k', ijk[2, :], 'Voxel k index', where='VoxelData')

    return bdata


def __get_xyz(img):
    if len(img.shape) == 4:
        # 4D-image
        i_len, j_len, k_len, t = img.shape
        affine = np.delete(np.delete(img.coordmap.affine, 3, axis=0), 3, axis=1)
    else:
        # 3D-image
        i_len, j_len, k_len = img.shape
        affine = img.coordmap.affine
    ijk = np.array(list(itertools.product(xrange(i_len),
                                          xrange(j_len),
                                          xrange(k_len),
                                          [1]))).T
    return np.dot(affine, ijk)[:-1]


def __load_mri(fpath):
    '''Load a MRI image.

    - Returns data as 2D array (sample x voxel)
    - Returns voxle xyz coordinates (3 x voxel)
    - Returns voxel ijk indexes (3 x voxel)
    - Data, xyz, and ijk are flattened by Fortran-like index order
    '''
    img = nipy.load_image(fpath)

    data = img.get_data()
    if data.ndim == 4:
        data = data.reshape(-1, data.shape[-1], order='F').T
        i_len, j_len, k_len, t = img.shape
        affine = np.delete(np.delete(img.coordmap.affine, 3, axis=0), 3, axis=1)
    elif data.ndim == 3:
        data = data.flatten(order='F')
        i_len, j_len, k_len = img.shape
        affine = img.coordmap.affine
    else:
        raise ValueError('Invalid shape.')

    ijk = np.array(np.unravel_index(np.arange(i_len * j_len * k_len),
                                    (i_len, j_len, k_len), order='F'))
    ijk_b = np.vstack([ijk, np.ones((1, i_len * j_len * k_len))])
    xyz_b = np.dot(affine, ijk_b)
    xyz = xyz_b[:-1]

    return data, xyz, ijk


if __name__ == '__main__':
    testdatapath = './testdata/fmriprep/bids'

    fmriprep = FmriprepData(testdatapath)
    print(fmriprep.data)

    label_mapper = {'stimulus_name' : {'n/a' : np.nan,
                                       'hoge' : 1,
                                       'fuga' : 2}}

    bdata_native = create_bdata_fmriprep(testdatapath, 'volume_native', label_mapper=label_mapper)
    bdata_standard = create_bdata_fmriprep(testdatapath, 'volume_standard', label_mapper=label_mapper)

    print(bdata_native.dataset.shape)
    print(bdata_standard.dataset.shape)

    bdata_native.save('test_fmriprep_native.h5')
    bdata_standard.save('test_fmriprep_standard.h5')
