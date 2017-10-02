'''Distributed computation module

This file is a part of BdPy.
'''


__all__ = ['DistComp']


import os


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
        os.remove(self.lockfile)


    def __lockfilename(self, comp_id):
        '''Return the lock file path'''
        return os.path.join(self.lockdir, comp_id + '.lock')
