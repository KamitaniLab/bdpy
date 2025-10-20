import os, sys
import re
import shutil
import json
from glob import glob
import traceback
from bdpy import makedir_ifnot
import subprocess


def show_collated_data(src, source_type='bids_daily', root_dir='./', bids_dir='bids', fmap=False):
    """
    Check existing file counts by parsing daily experimetal directories based on Yaml.
    :param src: Structured data information loaded from yaml file.
        - Based on the daily experiment data organization.
    :param source_type: Source data type. Currently only 'bids_daily' is supported.
    :param root_dir: Root directory of the daily experiment data
    :param bids_dir: BIDS directory name under each experiment date directory (e.g., 'bids', 'bids_fmap', etc.)
    :param fmap: Whether to check field map data
    """
    if not source_type == 'bids_daily':
        raise NotImplementedError('Source type %s not supported.' % source_type)
    
    for subject, sub_data in src.items():
        print('----------------------------------------')
        print(subject)

        # Reference anatomy
        src_anat = os.path.join(root_dir, sub_data['anat'])
        if not os.path.exists(src_anat):
            raise RuntimeError('Anatomy file not found: %s' % src_anat)

        # Functionals
        tasks = sub_data['func']
        for task, srcdata_dict in tasks.items():
            print('--------------------')
            print('{} includes {} experimental dates'.format(task, len(srcdata_dict.keys())))
            print('')

            # Prepare source data
            ses_num = 0
            run_num = 0
            for src_name, src_info in srcdata_dict.items():
                src_bids_path = os.path.join(root_dir, src_name, bids_dir)
                if not os.path.exists(src_bids_path):
                    raise RuntimeError('Not found BIDS directory: %s' % src_bids_path)
                else:
                    print("Parse BIDS directory:", src_bids_path)
                
                sessions = __parse_bids_dir(src_bids_path, data_info=src_info)
                ses_num += len(sessions)
                for ses in sessions:
                    run_num += len(ses['functionals'])

                    src_inplane = ses['inplane']
                    if not os.path.exists(src_inplane):
                        raise RuntimeError('Inplane image found: %s' % src_inplane)

                    if fmap:
                        src_fmap_dir = os.path.join(os.path.dirname(os.path.dirname(ses['inplane'])), 'fmap')
                        fmap_files = glob(os.path.join(src_fmap_dir, '*.nii.gz'))
                        if not fmap_files:
                            raise RuntimeError('No field map files found in %s' % src_fmap_dir)

            print('Total sessions: %d' % ses_num)
            print('Total runs: %d' % run_num)
            print('')

        return None


def makedata(src, source_type='bids_daily', output_dir='./output', root_dir='./', bids_dir='bids', 
             anatomy=True, inplane=True, fmap=False, functionals=True, run_deface=False,
             dry_run=False):
    '''
    Create BIDS dataset for OpenNeuro.
    :param src: Structured data information loaded from yaml file.
        - Based on the daily experiment data organization.
    :param source_type: Source data type. Currently only 'bids_daily' is supported.
    :param output_dir: Output directory for costructed BIDS dataset.
    :param root_dir: Root directory of the input daily experiment data
    :param bids_dir: BIDS directory name under each experiment date directory (e.g., 'bids', 'bids_fmap', etc.)
    :param anatomy: Whether to convert anatomy data
    :param inplane: Whether to convert inplane T2 data
    :param fmap: Whether to convert field map data
    :param functionals: Whether to convert functional data
    :param run_deface: Whether to run pydeface for anatomy data
        ! If pydeface doesn't work, set it to False.
    :param dry_run: If True, only print the operations without actual file copying
    '''

    if not source_type == 'bids_daily':
        raise NotImplementedError('Source type %s not supported.' % source_type)

    __create_dir(output_dir)
    __create_dir(os.path.join(output_dir, 'sourcedata'))
    __create_dir(os.path.join(output_dir, 'derivatives'))

    readme = os.path.join(output_dir, 'README')
    description = os.path.join(output_dir, 'dataset_description.json')
    if not os.path.exists(readme):
        print('Creating %s' % readme)
        with open(readme, 'w'):
            pass
    if not os.path.exists(description):
        print('Creating %s' % description)
        desc_init = {
            'Name': '',
            'License': '',
            'Authors': [],
            'Acknowledgements': '',
            'HowToAcknowledge': '',
            'Funding': [],
            'ReferencesAndLinks': [],
            'DatasetDOI': '',
            'BIDSVersion': '1.0.2',
        }
        with open(description, 'w') as f:
            json.dump(desc_init, f, indent=4)

    for subject, sub_data in src.items():
        print('----------------------------------------')
        print(subject)

        subject_dir = os.path.join(output_dir, subject)
        __create_dir(subject_dir)

        # Reference anatomy
        src_anat = os.path.join(root_dir, sub_data['anat'])
        trg_anat_dir = os.path.join(subject_dir, 'ses-anatomy', 'anat')
        trg_anat_raw = os.path.join(trg_anat_dir, '%s_ses-anatomy_T1w_raw.nii.gz' % subject)
        trg_anat_defaced = os.path.join(trg_anat_dir, '%s_ses-anatomy_T1w.nii.gz' % subject)
        __create_dir(trg_anat_dir)

        if anatomy:
            if not dry_run and not os.path.exists(trg_anat_defaced):
                ext = os.path.splitext(src_anat)[1]
                if ext == '.gz':
                    shutil.copy2(src_anat, trg_anat_raw)  # FIXME: convert to nii.gz
                else:
                    convert_command = 'mri_convert %s %s' % (src_anat, trg_anat_raw)
                    result = subprocess.run(convert_command.split(" "), capture_output=True, text=True)
                    if result.returncode != 0:
                        raise RuntimeError(f"{result.stdout}")
                    
                # Due to environment-dependent issues, it is possible to specify whether to run deface.
                if run_deface:
                    deface_command = 'pydeface %s --outfile %s' % (trg_anat_raw, trg_anat_defaced)
                    result = subprocess.run(deface_command.split(" "), capture_output=True, text=True)
                    if result.returncode != 0:
                        raise RuntimeError(f"{result.stdout}")

        # Functionals
        tasks = sub_data['func']

        if tasks is None:
            continue

        if isinstance(tasks, list):
            tasks = {'': tasks}

        print('%d task(s)' % len(tasks))
        print('')

        for task, srcdata_dict in tasks.items():
            print('--------------------')
            print('{} includes {} data'.format(task, len(srcdata_dict.keys())))
            print('')

            # Prepare source data
            task_sessions = []

            for src_name, src_info in srcdata_dict.items():
                src_bids_path = os.path.join(root_dir, src_name, bids_dir)
                print(src_bids_path)

                if not os.path.isdir(src_bids_path):
                    raise RuntimeError('Invalid BIDS directory: %s' % src_bids_path)

                sessions = __parse_bids_dir(src_bids_path, data_info=src_info)
                task_sessions.extend(sessions)

            print('Total sessions: %d' % len(task_sessions))
            print('Total runs: %d' % sum([len(ses['functionals']) for ses in task_sessions]))
            print('')

            # Copy files
            for i, ses in enumerate(task_sessions):
                session_label = 'ses-%s%02d' % (task, i +1)
                print('Session: %s' % session_label)

                session_dir = os.path.join(subject_dir, session_label)

                if not dry_run:
                    __create_dir(session_dir)
                    __create_dir(os.path.join(session_dir, 'anat'))
                    __create_dir(os.path.join(session_dir, 'func'))

                src_inplane = ses['inplane']
                rename_table = {
                    os.path.basename(src_inplane).split('_')[0]: subject,       # SUbject ID
                    os.path.basename(src_inplane).split('_')[1]: session_label, # Session label
                }

                # T2 inplane image
                if inplane:
                    src_inplane = ses['inplane']
                    if not src_inplane is None:
                        trg_inplane = os.path.join(session_dir, 'anat',
                                                __rename_file(os.path.basename(src_inplane).split('.')[0] + '.nii.gz',
                                                                rename=rename_table))
                        print('Copying\n  from: %s\n  to: %s' % (src_inplane, trg_inplane))
                        if not dry_run and not os.path.exists(trg_inplane):
                            ext = os.path.splitext(src_inplane)[1]
                            if ext == '.gz':
                                shutil.copy2(src_inplane, trg_inplane)
                            else:
                                convert_command = 'mri_convert %s %s' % (src_inplane, trg_inplane)
                                result = subprocess.run(convert_command.split(" "), capture_output=True, text=True)
                                if result.returncode != 0:
                                    raise RuntimeError(f"{result.stdout}")

                # Field map
                if fmap:
                    src_fmap_dir = os.path.join(os.path.dirname(os.path.dirname(ses['inplane'])), 'fmap')
                    trg_fmap_dir = os.path.join(session_dir, 'fmap')
                    if not dry_run:
                        __create_dir(os.path.join(session_dir, 'fmap'))

                    for f in glob(os.path.join(src_fmap_dir, '*.nii.gz')):
                        src_f = f
                        trg_f = os.path.join(
                            trg_fmap_dir,
                            __rename_file(os.path.basename(src_f).split('.')[0] + '.nii.gz', rename=rename_table)
                        )
                        if dry_run:
                            print('Copying\n  from: %s\n  to: %s' % (src_f, trg_f))
                        elif not os.path.exists(trg_f):
                            print('Copying\n  from: %s\n  to: %s' % (src_f, trg_f))
                            ext = os.path.splitext(src_f)[1]
                            if ext == '.gz':
                                shutil.copy2(src_f, trg_f)
                            else:
                                convert_command = 'mri_convert %s %s' % (src_f, trg_f)
                                result = subprocess.run(convert_command.split(" "), capture_output=True, text=True)
                                if result.returncode != 0:
                                    raise RuntimeError(f"{result.stdout}")

                    for f in glob(os.path.join(src_fmap_dir, '*.json')):
                        src_f = f
                        trg_f = os.path.join(
                            trg_fmap_dir,
                            __rename_file(os.path.basename(src_f).split('.')[0] + '.json', rename=rename_table)
                        )
                        if dry_run:
                            print('Copying\n  from: %s\n  to: %s' % (src_f, trg_f))
                        elif not os.path.exists(trg_f):
                            print('Copying\n  from: %s\n  to: %s' % (src_f, trg_f))
                            with open(src_f, 'r') as f:
                                js = json.load(f)
                            if 'IntendedFor' in js:
                                fs = []
                                for fi, f in enumerate(js['IntendedFor']):
                                    # # IntendedFor includes the original number of runs, so when using exclude runs, you need to reduce the number.
                                    if fi < len(ses['functionals']): 
                                        f = __rename_file(f, rename=rename_table)
                                        fs.append(f)
                                js['IntendedFor'] = fs
                            with open(trg_f, 'w') as f:
                                json.dump(js, f, indent=4)

                # Functionals
                if functionals:
                    for j, run in enumerate(ses['functionals']):
                        src_bold = run['bold']
                        src_bold_json = run['bold_json']
                        src_event = run['event']

                        # File renaming
                        rename_table = {
                            os.path.basename(src_bold).split('_')[0]: subject,       # SUbject ID
                            os.path.basename(src_bold).split('_')[1]: session_label, # Session label
                                    }

                        # Fix run label
                        run_label = re.match('.*_run-(\d+)_.*', os.path.basename(src_bold)).group(1)
                        if (j + 1) != int(run_label):
                            print('Fix run label: run-%s --> run-%02d' % (run_label, j + 1))
                            rename_table.update({'run-%s' % run_label: 'run-%02d' % (j + 1)})

                        trg_bold = os.path.join(session_dir, 'func',
                                                __rename_file(os.path.basename(src_bold).split('.')[0] + '.nii.gz',
                                                            rename=rename_table))
                        trg_bold_json = os.path.join(session_dir, 'func',
                                                    __rename_file(os.path.basename(src_bold_json), rename=rename_table))
                        trg_event = os.path.join(session_dir, 'func',
                                                __rename_file(os.path.basename(src_event), rename=rename_table))

                        print('Copying\n  from: %s\n  to: %s' % (src_bold, trg_bold))
                        print('Copying\n  from: %s\n  to: %s' % (src_bold_json, trg_bold_json))
                        print('Copying\n  from: %s\n  to: %s' % (src_event, trg_event))
                        if not dry_run and not os.path.exists(trg_bold):
                            ext = os.path.splitext(src_bold)[1]
                            if ext == '.gz':
                                shutil.copy2(src_bold, trg_bold)
                            else:
                                convert_command = 'mri_convert %s %s' % (src_bold, trg_bold)
                                result = subprocess.run(convert_command.split(" "), capture_output=True, text=True)
                                if result.returncode != 0:
                                    raise RuntimeError(f"{result.stdout}")

                            shutil.copy2(src_bold_json, trg_bold_json)
                            shutil.copy2(src_event, trg_event)

    return None


def __parse_bids_dir(dpath, data_info=None):
    """
    Parse BIDS directory for a specific experimental date and obtain session and file information.
    :param dpath: BIDS directory path
    :param data_info: Structured data information for a specific experimental date (sessions, excluded runs, etc.)
    """
    print('BIDS directory: %s' % dpath)

    sub_dirs = glob(os.path.join(dpath, 'sub-*'))
    if len(sub_dirs) != 1:
        raise RuntimeError('Unsupported BIDS data (invalid number of subjects: %d)' % len(sub_dirs))
    sub_dir = os.path.join(dpath, sub_dirs[0])
    print('Subject directory: %s' % sub_dir)

    ses_dirs = sorted(glob(os.path.join(sub_dirs[0], 'ses-*', 'func')))
    print('%d func session(s) found' % len(ses_dirs))

    sessions = []
    unskip_session_counter = -1
    for i, ses_dir in enumerate(ses_dirs):
        session_id = i + 1

        # Session selection
        if (not data_info is None) and 'ses' in data_info:
            if session_id not in data_info['ses']:
                print('Skipping session %02d' % (i + 1))
                continue
        
        # T2 inplane image
        inplane_file = __aggregate_mri_files(os.path.join(ses_dir, '../anat'), mri_filetype='nii') + __aggregate_mri_files(os.path.join(ses_dir, '../anat'), mri_filetype='nii.gz')
        if len(inplane_file) != 1:
            raise RuntimeError('Invalid inplane anatomy')
        inplane_file = inplane_file[0]

        # Functionals
        run_files = __aggregate_runs(ses_dir, mri_filetype='nii') + __aggregate_runs(ses_dir, mri_filetype='nii.gz')
        print('Ses %02d: %d run(s) found' % (session_id, len(run_files)))
        #print(run_files)

        # Run selection
        run_files_keep = []
        if (not data_info is None) and 'discard_run' in data_info:
            for j, rf in enumerate(run_files):
                run_id = j + 1
                if run_id in data_info["discard_run"][unskip_session_counter]:
                    print("Discard run:", run_id)
                else:
                    run_files_keep.append(rf)
        else:
            run_files_keep = run_files

        print("Use %d run(s)" % (len(run_files_keep)))

        sessions.append({'inplane': inplane_file,
                         'functionals': run_files_keep})
        unskip_session_counter += 1

    print('')

    return sessions


def __aggregate_runs(dpath, mri_filetype='nii'):
    mri_files = __aggregate_mri_files(dpath, mri_filetype=mri_filetype)

    run_files = []
    for mri_file in mri_files:
        basename = os.path.basename(mri_file)

        # Json file
        json_file = os.path.join(dpath, basename.replace('_bold.' + mri_filetype, '_bold.json'))
        if not os.path.isfile(json_file):
            json_file = None

        # Task event file
        task_event_file = os.path.join(dpath, basename.replace('_bold.' + mri_filetype, '_events.tsv'))
        if not os.path.isfile(task_event_file):
            task_event_file = None

        scan_files = sorted(glob(os.path.join(dpath, basename + '*')))

        run_files.append({'bold': mri_file,
                          'bold_json': json_file,
                          'event': task_event_file})

    return run_files


def __aggregate_mri_files(dpath, mri_filetype='nii'):
    mri_files = glob(os.path.join(dpath, '*.' + mri_filetype))
    return mri_files


def __rename_file(fname, rename={}):
    for before, after in rename.items():
        fname = fname.replace(before, after)
    return fname


def __create_dir(dirpath):
    if not os.path.exists(dirpath):        
        print('Creating %s' % dirpath)
        os.makedirs(dirpath)
    return None