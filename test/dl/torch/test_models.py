import unittest

import torch
import torch.nn as nn

from bdpy.dl.torch import models


class MockModule(nn.Module):
    def __init__(self):
        super(MockModule, self).__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layers = nn.Sequential(
            nn.Conv2d(1, 1, 3),
            nn.Conv2d(1, 1, 3),
            nn.Module(),
            nn.Sequential(
                nn.Conv2d(1, 1, 4),
                nn.Conv2d(1, 1, 8),
            )
        )
        inner_network = self.layers[-2]
        inner_network.features = nn.Sequential(
            nn.Conv2d(1, 1, 5),
            nn.Conv2d(1, 1, 5)
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


class TestParseLayerName(unittest.TestCase):
    def setUp(self):
        self.mock = MockModule()
        self.accessors = [
            {'name': 'layer1', 'type': nn.Linear, 'attrs': {'in_features': 10, 'out_features': 10}},
            {'name': 'layers[0]', 'type': nn.Conv2d, 'attrs': {'kernel_size': (3, 3)}},
            {'name': 'layers[1]', 'type': nn.Conv2d, 'attrs': {'kernel_size': (3, 3)}},
            {'name': 'layers[2].features[0]', 'type': nn.Conv2d, 'attrs': {'kernel_size': (5, 5)}},
            {'name': 'layers[3][0]', 'type': nn.Conv2d, 'attrs': {'kernel_size': (4, 4)}},
            {'name': 'layers[3][1]', 'type': nn.Conv2d, 'attrs': {'kernel_size': (8, 8)}}
        ]

    def test_parse_layer_name(self):
        for accessor in self.accessors:
            layer = models._parse_layer_name(self.mock, accessor['name'])
            self.assertIsInstance(layer, accessor['type'])
            for attr, value in accessor['attrs'].items():
                self.assertEqual(getattr(layer, attr), value)

        # Test non-existing layer access
        self.assertRaises(
            ValueError, models._parse_layer_name, self.mock, 'not_existing_layer')
        # Test invalid layer access
        self.assertRaises(
            ValueError, models._parse_layer_name, self.mock, 'layers["key"]')


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


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParseLayerName)
    unittest.TextTestRunner(verbosity=2).run(suite)
