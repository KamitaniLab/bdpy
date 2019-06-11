'''Distributed computation module

This file is a part of BdPy.
'''


__all__ = ['DistComp']


import os
import warnings


class DistComp(object):
    '''Distributed computation class'''


    def __init__(self, comp_id=None, lockdir='tmp'):
        self.lockdir = lockdir
        self.comp_id = comp_id

        self.lockfile = self.__lockfilename(self.comp_id) if self.comp_id != None else None


    def islocked(self):
        if os.path.isfile(self.lockfile):
            return True
        else:
            return False
        

    def lock(self):
        with open(self.lockfile, 'w'):
            pass

        
    def unlock(self):
        try:
            os.remove(self.lockfile)
        except OSError:
            warnings.warn('Failed to unlock the computation. Possibly double running.')


    def islocked_lock(self):
        is_locked = os.path.isfile(self.lockfile)
        if not is_locked:
            with open(self.lockfile, 'w'):
                pass

        return is_locked


    def __lockfilename(self, comp_id):
        '''Return the lock file path'''
        return os.path.join(self.lockdir, comp_id + '.lock')
