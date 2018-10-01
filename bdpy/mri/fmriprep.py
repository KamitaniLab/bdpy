'''Utilities for fmriprep.'''


import os
import glob
import re
from collections import OrderedDict


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
            self.__data.update({sbj: {}})
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
        return sessions

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
                 'confounds' : os.path.join('derivatives', subject, session, 'func', conf)}
                for nat, std, conf in zip(prep_vol_native, prep_vol_standard, confounds)]

        return runs

    def __get_task_event_files(self):
        for sbj, sbjdata in self.__data.items():
            for ses, sesdata in sbjdata.items():
                raw_func_dir = os.path.join(self.__datapath, sbj, ses, 'func')
                #print(raw_func_dir)
                for run in sesdata:
                    #print(run)
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


if __name__ == '__main__':
    testdatapath = './testdata/fmriprep/bids'

    fmriprep = FmriprepData(testdatapath)
    print(fmriprep.data)
