'''Tests for distcomp'''

import unittest

import os
import tempfile

from bdpy.distcomp import DistComp


class TestDistComp(unittest.TestCase):
    def test_distcomp_file(self):
        with tempfile.TemporaryDirectory() as lockdir:
            comp_id = 'test-distcomp-fs'

            # init
            distcomp = DistComp(lockdir=lockdir, comp_id=comp_id)
            self.assertTrue(os.path.isdir(lockdir))
            self.assertFalse(distcomp.islocked())

            # lock
            distcomp.lock()
            self.assertTrue(os.path.isfile(os.path.join(lockdir, comp_id + '.lock')))
            self.assertTrue(distcomp.islocked())

            # unlock
            distcomp.unlock()
            self.assertFalse(os.path.isfile(os.path.join(lockdir, comp_id + '.lock')))
            self.assertFalse(distcomp.islocked())

            # islocked_lock
            distcomp.islocked_lock()
            self.assertTrue(os.path.isfile(os.path.join(lockdir, comp_id + '.lock')))
            self.assertTrue(distcomp.islocked())

    def test_distcomp_sqlite3(self):
        with tempfile.TemporaryDirectory() as lockdir:
            db_path = os.path.join(lockdir, 'distcomp.db')
            comp_id = 'test-distcomp-sqlite3-1'

            # init
            distcomp = DistComp(backend='sqlite3', db_path=db_path)
            self.assertTrue(os.path.isfile(db_path))
            self.assertFalse(distcomp.islocked(comp_id))

            # lock
            distcomp.lock(comp_id)
            self.assertTrue(distcomp.islocked(comp_id))

            # unlock
            distcomp.unlock(comp_id)
            self.assertFalse(distcomp.islocked(comp_id))

            # islocked_lock
            with self.assertRaises(NotImplementedError):
                distcomp.islocked_lock(comp_id)


if __name__ == '__main__':
    unittest.main()