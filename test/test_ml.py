'''Tests for ml'''


import os
import unittest
import shutil

import numpy as np

from bdpy.dataform import load_array
from bdpy.ml import ModelTraining
from bdpy.distcomp import DistComp

from sklearn.linear_model import LinearRegression
from fastl2lir import FastL2LiR


class TestUtil(unittest.TestCase):
    def test_ModelTraining_sklern_pickle_nochunk(self):
        # Init
        results_dir_root = './results/ml'
        if os.path.exists(results_dir_root):
            shutil.rmtree(results_dir_root)
        os.makedirs(results_dir_root)

        # Data setup
        X = np.random.rand(100, 500)
        Y = np.random.rand(100, 50)

        # Test
        model = LinearRegression()

        train = ModelTraining(model, X, Y)
        train.id = os.path.splitext(os.path.basename(__file__))[0] + '-lir-nochunk'
        train.save_path = os.path.join(results_dir_root, 'lir-nochunk', 'model.pkl')

        train.run()

        self.assertTrue(os.path.isfile(os.path.join(results_dir_root, 'lir-nochunk', 'model.pkl')))

    def test_ModelTraining_fastl2lir_pickle_nochunk(self):
        # Init
        results_dir_root = './results/ml'
        if os.path.exists(results_dir_root):
            shutil.rmtree(results_dir_root)
        os.makedirs(results_dir_root)

        # Data setup
        X = np.random.rand(100, 500)
        Y = np.random.rand(100, 50)

        # Test
        model = FastL2LiR()
        model_param = {'alpha'  : 100,
                       'n_feat' : 100}

        train = ModelTraining(model, X, Y)
        train.id = os.path.splitext(os.path.basename(__file__))[0] + '-fastl2lir-nochunk'
        train.model_parameters = model_param
        train.save_path = os.path.join(results_dir_root, 'fastl2lir-nochunk', 'model.pkl')

        train.run()

        self.assertTrue(os.path.isfile(os.path.join(results_dir_root, 'fastl2lir-nochunk', 'model.pkl')))

    def test_ModelTraining_fastl2lir_bdmodel_nochunk(self):
        # Init
        results_dir_root = './results/ml'
        if os.path.exists(results_dir_root):
            shutil.rmtree(results_dir_root)
        os.makedirs(results_dir_root)

        # Data setup
        X = np.random.rand(100, 500)
        Y = np.random.rand(100, 50)

        # Test
        model = FastL2LiR()
        model_param = {'alpha'  : 100,
                       'n_feat' : 100}

        train = ModelTraining(model, X, Y)
        train.id = os.path.splitext(os.path.basename(__file__))[0] + '-fastl2lir-nochunk-bdmodel'
        train.model_parameters = model_param
        train.dtype = np.float32
        train.save_format = 'bdmodel'
        train.save_path = os.path.join(results_dir_root, 'fastl2lir-nochunk-bdmodel', 'model')

        train.run()

        self.assertTrue(os.path.isfile(os.path.join(results_dir_root, 'fastl2lir-nochunk-bdmodel', 'model', 'W.mat')))
        self.assertTrue(os.path.isfile(os.path.join(results_dir_root, 'fastl2lir-nochunk-bdmodel', 'model', 'b.mat')))

        W = load_array(os.path.join(results_dir_root, 'fastl2lir-nochunk-bdmodel', 'model', 'W.mat'), key='W')
        b = load_array(os.path.join(results_dir_root, 'fastl2lir-nochunk-bdmodel', 'model', 'b.mat'), key='b')

        self.assertEqual(W.shape, (500, 50))
        self.assertEqual(b.shape, (1, 50))

    def test_ModelTraining_fastl2lir_bdmodel_chunk(self):
        # Init
        results_dir_root = './results/ml'
        if os.path.exists(results_dir_root):
            shutil.rmtree(results_dir_root)
        os.makedirs(results_dir_root)

        # Data setup
        X = np.random.rand(100, 500)
        Y = np.random.rand(100, 8, 4, 4)

        # Test
        model = FastL2LiR()
        model_param = {'alpha'  : 100,
                       'n_feat' : 100}

        train = ModelTraining(model, X, Y)
        train.id = os.path.splitext(os.path.basename(__file__))[0] + '-fastl2lir-chunk-bdmodel'
        train.model_parameters = model_param
        train.dtype = np.float32
        train.chunk_axis = 1
        train.save_format = 'bdmodel'
        train.save_path = os.path.join(results_dir_root, 'fastl2lir-chunk-bdmodel', 'model')

        train.run()

        for i in range(Y.shape[train.chunk_axis]):
            self.assertTrue(os.path.isfile(os.path.join(results_dir_root, 'fastl2lir-chunk-bdmodel', 'model', 'W', '%08d.mat' % i)))
            self.assertTrue(os.path.isfile(os.path.join(results_dir_root, 'fastl2lir-chunk-bdmodel', 'model', 'b', '%08d.mat' % i)))

            W = load_array(os.path.join(results_dir_root, 'fastl2lir-chunk-bdmodel', 'model', 'W', '%08d.mat' % i), key='W')
            b = load_array(os.path.join(results_dir_root, 'fastl2lir-chunk-bdmodel', 'model', 'b', '%08d.mat' % i), key='b')

            self.assertEqual(W.shape, (500, 1, 4, 4))
            self.assertEqual(b.shape, (1, 1, 4, 4))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtil)
    unittest.TextTestRunner(verbosity=2).run(suite)
