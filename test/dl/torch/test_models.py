import unittest

import torch
import torch.nn as nn

from bdpy.dl.torch import models


class MockModule(nn.Module):
    def __init__(self):
        super(MockModule, self).__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layers = nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.Conv2d(3, 3, 3)
        )


class TestLayerMap(unittest.TestCase):
    def setUp(self):
        self.kv_pairs = [
            {'net': 'vgg19', 'payload': {'key': 'fc6', 'value': 'classifier[0]'}},
            {'net': 'vgg19', 'payload': {'key': 'conv5_4', 'value': 'features[34]'}},
            {'net': 'alexnet', 'payload': {'key': 'fc6', 'value': 'classifier[0]'}},
            {'net': 'alexnet', 'payload': {'key': 'conv5', 'value': 'features[12]'}}
        ]

    def test_layer_map(self):
        for kv_pair in self.kv_pairs:
            expected = kv_pair['payload']
            output = models.layer_map(kv_pair['net'])
            self.assertIsInstance(output, dict)
            self.assertEqual(output[expected['key']], expected['value'])

    def test_parse_layer_name(self):
        mock = MockModule()
        self.assertIsInstance(models._parse_layer_name(mock, 'layer1'), nn.Linear)
        self.assertIsInstance(models._parse_layer_name(mock, 'layers[0]'), nn.Conv2d)
        self.assertIsInstance(models._parse_layer_name(mock, 'layers[1]'), nn.Conv2d)


class TestVGG19(unittest.TestCase):
    def setUp(self):
        self.input_shape = (1, 3, 224, 224)
        self.model = models.VGG19()

    def test_forward(self):
        x = torch.rand(self.input_shape)
        output = self.model(x)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 1000))

    def test_layer_access(self):
        layer_names = models.layer_map('vgg19').values()
        for layer_name in layer_names:
            self.assertIsInstance(
                models._parse_layer_name(self.model, layer_name), nn.Module)


class TestAlexNet(unittest.TestCase):
    def setUp(self):
        self.input_shape = (1, 3, 224, 224)
        self.model = models.AlexNet()

    def test_forward(self):
        x = torch.rand(self.input_shape)
        output = self.model(x)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 1000))

    def test_layer_access(self):
        layer_names = models.layer_map('alexnet').values()
        for layer_name in layer_names:
            self.assertIsInstance(
                models._parse_layer_name(self.model, layer_name), nn.Module)


class TestAlexNetGenerator(unittest.TestCase):
    def setUp(self):
        self.input_shape = (1, 4096)
        self.model = models.AlexNetGenerator()

    def test_forward(self):
        x = torch.rand(self.input_shape)
        output = self.model(x)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 3, 256, 256))
