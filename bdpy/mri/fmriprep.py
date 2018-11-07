'''Utilities for fmriprep.'''


import csv
import glob
import itertools
import os
import re
from collections import OrderedDict

import numpy as np
import nipy
import pandas as pd

import bdpy


class FmriprepData(object):
    '''FMRIPREP data class.'''

    def __init__(self, datapath=None):
        self.__datapath = datapath
        self.__data = OrderedDict()

        if self.__datapath is not None:
            self.__parse_data()
            self.__get_task_event_files()

        return None

    @property
    def data(self):
        return self.__data

    # Private methods --------------------------------------------------------

    def __parse_data(self):
        #print('Data path: %s' % self.__datapath)

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

        prep_vol_native = []    # List of preprocessed EPI files in native space (T1w)
        prep_vol_standard = []  # List of preprocessed EPI files in MNI space
        confounds = []          # List of confound files

        for f in os.listdir(funcpath):
            if os.path.isdir(os.path.join(funcpath, f)):
                continue

            [fname, fext] = os.path.splitext(f)

            if fext == '.gii':
                # Surface data not supported yet.
                continue

            # Get Motion-corrected EPI files in native T1w (*_bold_space-T1w_preproc.nii.gz)
            m = re.search('.*_bold_space-T1w_preproc\.nii\.gz$', f)
            if m:
                prep_vol_native.append(f)

            # Get Motion-corrected EPI files in MNI space (*_bold_space-MNI152NLin2009cAsym_preproc\.nii\.gz)
            m = re.search('.*_bold_space-MNI152NLin2009cAsym_preproc\.nii\.gz$', f)
            if m:
                prep_vol_standard.append(f)

            # Get confound file (*_bold_confounds.tsv)
            m = re.search('.*_bold_confounds\.tsv$', f)
            if m:
                confounds.append(f)

        prep_vol_native.sort()
        prep_vol_standard.sort()
        confounds.sort()

        # TODO: add run num check

        runs = [{'volume_native' : os.path.join('derivatives', 'fmriprep', 'fmriprep', subject, session, 'func', nat),
                 'volume_standard' : os.path.join('derivatives', 'fmriprep', 'fmriprep', subject, session, 'func', std),
                 'confounds' : os.path.join('derivatives', 'fmriprep', 'fmriprep', subject, session, 'func', conf)}
                for nat, std, conf in zip(prep_vol_native, prep_vol_standard, confounds)]

        return runs

    def __get_task_event_files(self):
        for sbj, sbjdata in self.__data.items():
            for ses, sesdata in sbjdata.items():
                raw_func_dir = os.path.join(self.__datapath, sbj, ses, 'func')
                for run in sesdata:
                    # Get run label
                    m = re.search('.*_(run-.*)_bold_.*', run['confounds'])
                    if m:
                        run_label = m.group(1)
                    else:
                        raise RuntimeError('Run not found!')
                    # Get task event file
                    event_file_name_glob = '%s_%s_task-*_%s_events.tsv' % (sbj, ses, run_label)
                    event_file_list = glob.glob(os.path.join(raw_func_dir, event_file_name_glob))
                    if len(event_file_list) != 1:
                        raise RuntimeError('Something is wrong on task event files.')
                    event_file = event_file_list[0].replace(self.__datapath + '/', '')
                    # Add the task event file in data
                    run.update({'task_event_file': event_file})
        return None


def create_bdata_fmriprep(dpath, data_mode='volume_standard', label_mapper=None):
    '''Create BData from FMRIPREP outputs.

    Parameters
    ----------
    dpath : str
        Path to a BIDS data directory.
    data_mode: {'volume_standard', 'volume_native'}
        Data to be loaded.
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
    fmriprep = FmriprepData(dpath)

    for sbj, sbjdata in fmriprep.data.items():
        print('----------------------------------------')
        print('Subject: %s\n' % sbj)

        bdata = __create_bdata_fmriprep_subject(sbjdata, data_mode, data_path=dpath, label_mapper=label_mapper_dict)

    return bdata


def __create_bdata_fmriprep_subject(subject_data, data_mode, data_path='./', label_mapper={}):
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
            print('EPI:             %s' % epi)
            print('Task event file: %s' % event_file)
            print('Confounds file:  %s' % confounds_file)

            mp_label = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']

            # Load volume
            data, xyz_run, ijk_run = __load_mri(os.path.join(data_path, epi))

            braindata_list.append(data)
            xyz = xyz_run
            ijk = ijk_run

            num_vol = data.shape[0]

            # Load motion parameters (and the other confounds)
            conf_pd = pd.read_csv(os.path.join(data_path, confounds_file), delimiter='\t')

            mp = np.hstack([np.c_[conf_pd[a]] for a in mp_label])
            motionparam_list.append(mp)

            # Load task event file
            event_file = os.path.join(data_path, run['task_event_file'])
            events = pd.read_csv(event_file, delimiter='\t')

            # Check time length
            tlen_event = events.tail(1)['onset'].values[0] + events.tail(1)['duration'].values[0]
            n_sample = data.shape[0]

            img = nipy.load_image(os.path.join(data_path, epi))
            tr = img.coordmap.affine[3, 3]

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
                #print(blocks)

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

                #import pdb; pdb.set_trace()

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

    #import pdb; pdb.set_trace()

    # Create BData (one subject, one file)
    bdata = bdpy.BData()

    bdata.add(braindata, 'VoxelData')
    bdata.add(ses_label, 'Session')
    bdata.add(run_label, 'Run')
    bdata.add(block_label, 'Block')
    bdata.add(labels_label, 'Label')
    bdata.add(motionparam, 'MotionParameter')
    bdata.add_metadata('MotionParameter_x', [1, 0, 0, 0, 0, 0], 'Motion parameter: x translation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_y', [0, 1, 0, 0, 0, 0], 'Motion parameter: y translation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_z', [0, 0, 1, 0, 0, 0], 'Motion parameter: z translation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_rot_x', [0, 0, 0, 1, 0, 0], 'Motion parameter: x rotation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_rot_y', [0, 0, 0, 0, 1, 0], 'Motion parameter: y rotation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_rot_z', [0, 0, 0, 0, 0, 1], 'Motion parameter: z rotation', where='MotionParameter')

    for i, col in enumerate(cols):
        metadata_vec = np.empty((len(cols),))
        metadata_vec[:] = np.nan
        metadata_vec[i] = 1
        bdata.add_metadata(col, metadata_vec, 'Label %s' % col, where='Label')

    bdata.add_metadata('voxel_x', xyz[0, :], 'Voxel x coordinate', where='VoxelData')
    bdata.add_metadata('voxel_y', xyz[1, :], 'Voxel y coordinate', where='VoxelData')
    bdata.add_metadata('voxel_z', xyz[2, :], 'Voxel z coordinate', where='VoxelData')

    bdata.add_metadata('voxel_i', ijk[0, :], 'Voxel i index', where='VoxelData')
    bdata.add_metadata('voxel_j', ijk[1, :], 'Voxel j index', where='VoxelData')
    bdata.add_metadata('voxel_k', ijk[2, :], 'Voxel k index', where='VoxelData')

    # TODO: add voxel ijk

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
