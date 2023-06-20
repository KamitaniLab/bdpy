# coding: utf-8
'''Tests for ml'''


import os
import unittest
import shutil
import pickle
import tempfile

import numpy as np

from bdpy.dataform import load_array
from bdpy.ml import ModelTraining, ModelTest

from sklearn.linear_model import LinearRegression
from fastl2lir import FastL2LiR


def _make_id(key: str) -> str:
    return os.path.splitext(os.path.basename(__file__))[0] + f'-{key}'


class TestModelTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.results_dir_root = tempfile.TemporaryDirectory()
        self.X = np.random.rand(100, 500)
        self.Y1dim = np.random.rand(100, 50)
        self.Y4dim = np.random.rand(100, 8, 4, 4)

    def tearDown(self) -> None:
        self.results_dir_root.cleanup()

    def test_sklern_pickle_nochunk(self):
        X = self.X
        Y = self.Y1dim

        key = 'lir-nochunk'
        model = LinearRegression()

        train = ModelTraining(model, X, Y)
        train.id = _make_id(key)
        train.save_path = os.path.join(self.results_dir_root.name, key)

        train.run()

        self.assertTrue(os.path.isfile(os.path.join(train.save_path, 'model.pkl.gz')))

    def test_fastl2lir_pickle_nochunk(self):
        X = self.X
        Y = self.Y1dim

        key = 'fastl2lir-nochunk'
        model = FastL2LiR()
        model_parameters = {'alpha': 100, 'n_feat': 100}

        train = ModelTraining(model, X, Y)
        train.id = _make_id(key)
        train.model_parameters = model_parameters
        train.save_path = os.path.join(self.results_dir_root.name, key)

        train.run()

        self.assertTrue(os.path.isfile(os.path.join(train.save_path, 'model.pkl.gz')))

    def test_fastl2lir_bdmodel_nochunk(self):
        X = self.X
        Y = self.Y1dim

        key = 'fastl2lir-nochunk-bdmodel'
        model = FastL2LiR()
        model_parameters = {'alpha': 100, 'n_feat': 100}

        train = ModelTraining(model, X, Y)
        train.id = _make_id(key)
        train.model_parameters = model_parameters
        train.dtype = np.float32
        train.save_format = 'bdmodel'
        train.save_path = os.path.join(self.results_dir_root.name, key, 'model')

        train.run()

        self.assertTrue(os.path.isfile(os.path.join(train.save_path, 'W.mat')))
        self.assertTrue(os.path.isfile(os.path.join(train.save_path, 'b.mat')))

        W = load_array(os.path.join(train.save_path, 'W.mat'), key='W')
        b = load_array(os.path.join(train.save_path, 'b.mat'), key='b')

        self.assertEqual(W.shape, (500, 50))
        self.assertEqual(b.shape, (1, 50))

    def test_fastl2lir_bdmodel_chunk(self):
        X = self.X
        Y = self.Y4dim

        key = 'fastl2lir-chunk-bdmodel'
        model = FastL2LiR()
        model_parameters = {'alpha': 100, 'n_feat': 100}

        train = ModelTraining(model, X, Y)
        train.id = _make_id(key)
        train.model_parameters = model_parameters
        train.dtype = np.float32
        train.chunk_axis = 1
        train.save_format = 'bdmodel'
        train.save_path = os.path.join(self.results_dir_root.name, key, 'model')

        train.run()

        for i in range(Y.shape[train.chunk_axis]):
            self.assertTrue(os.path.isfile(os.path.join(train.save_path, 'W', '%08d.mat' % i)))
            self.assertTrue(os.path.isfile(os.path.join(train.save_path, 'b', '%08d.mat' % i)))

            W = load_array(os.path.join(train.save_path, 'W', '%08d.mat' % i), key='W')
            b = load_array(os.path.join(train.save_path, 'b', '%08d.mat' % i), key='b')

            self.assertEqual(W.shape, (500, 1, 4, 4))
            self.assertEqual(b.shape, (1, 1, 4, 4))


class TestModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_models_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir, 'data', 'test_models'))
        self.X = np.random.rand(30, 500)
        self.batch_shape = (30,)

    def test_sklearn_nochunk_pkl(self):
        key = 'lir-nochunk-pkl'
        X = self.X

        model_path = os.path.join(self.test_models_path, key, 'model.pkl.gz')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)['model']

        test = ModelTest(model, X)
        y_pred = test.run()
        self.assertEqual(y_pred.shape, (30, 50))

    def test_sklearn_nochunk_pkl_modelpath(self):
        key = 'lir-nochunk-pkl'
        X = self.X

        model = LinearRegression()
        model_path = os.path.join(self.test_models_path, key, 'model.pkl.gz')

        test = ModelTest(model, X)
        test.model_path = model_path
        y_pred = test.run()
        self.assertEqual(y_pred.shape, (30, 50))

    def test_fastl2lir_nochunk_pkl_modelpath(self):
        key = 'fastl2lir-nochunk-pkl'
        X = self.X

        model = FastL2LiR()
        model_path = os.path.join(self.test_models_path, key, 'model.pkl.gz')

        test = ModelTest(model, X)
        test.model_path = model_path
        y_pred = test.run()
        self.assertEqual(y_pred.shape, (30, 50))

    def test_fastl2lir_chunk_pkl_modelpath(self):
        key = 'fastl2lir-chunk-pkl'
        X = self.X

        model = FastL2LiR()
        model_path = os.path.join(self.test_models_path, key)

        test = ModelTest(model, X)
        test.model_path = model_path
        test.chunk_axis = 1
        y_pred = test.run()
        self.assertEqual(y_pred.shape, (30, 8, 4, 4))

    def test_fastl2lir_nochunk_bd_modelpath(self):
        key = 'fastl2lir-nochunk-bd'
        X = self.X

        model = FastL2LiR()
        model_path = os.path.join(self.test_models_path, key)

        test = ModelTest(model, X)
        test.model_path = model_path
        test.model_format = 'bdmodel'
        y_pred = test.run()
        self.assertEqual(y_pred.shape, (30, 50))

    def test_fastl2lir_chunk_bd_modelpath(self):
        key = 'fastl2lir-chunk-bd'
        X = self.X

        model = FastL2LiR()
        model_path = os.path.join(self.test_models_path, key)

        test = ModelTest(model, X)
        test.model_path = model_path
        test.model_format = 'bdmodel'
        test.chunk_axis = 1
        y_pred = test.run()
        self.assertEqual(y_pred.shape, (30, 8, 4, 4))


if __name__ == '__main__':
    unittest.main()