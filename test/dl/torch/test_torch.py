import unittest

import numpy as np
import torch
import torch.nn as nn

from bdpy.dl.torch import torch as bdtorch


class MockInnerModule(nn.Module):
    def __init__(self):
        super(MockInnerModule, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(3, 4),
            nn.Linear(4, 5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x


class MockModule(nn.Module):
    def __init__(self):
        super(MockModule, self).__init__()
        self.layer1 = nn.Linear(10, 1)
        self.layers = nn.Sequential(
            nn.Linear(1, 2),
            nn.Linear(2, 3)
        )
        self.out = MockInnerModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layers(x)
        x = self.out(x)
        return x


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = MockModule()
        self.input_tensor = np.random.random(size=(10,)).astype(np.float32)
        self.layer_list = [
            {'map': {'alias': 'L1', 'entity': 'layer1'}, 'shape': (1, 1)},
            {'map': {'alias': 'L2', 'entity': 'layers[0]'}, 'shape': (1, 2)},
            {'map': {'alias': 'L3', 'entity': 'layers[1]'}, 'shape': (1, 3)},
            {'map': {'alias': 'L4', 'entity': 'out.features[0]'}, 'shape': (1, 4)},
            {'map': {'alias': 'L5', 'entity': 'out.features[1]'}, 'shape': (1, 5)}
        ]

    def test_run(self):
        self.encoder.eval()
        layer_to_shape = {payload['map']['entity']: payload['shape'] for payload in self.layer_list}
        extractor = bdtorch.FeatureExtractor(self.encoder, layer_to_shape.keys(), detach=True)
        features = extractor.run(self.input_tensor)
        for layer, shape in layer_to_shape.items():
            self.assertEqual(features[layer].shape, shape)

    def test_run_with_layer_map(self):
        self.encoder.eval()
        layer_to_shape = {payload['map']['alias']: payload['shape'] for payload in self.layer_list}
        layer_map = {payload['map']['alias']: payload['map']['entity'] for payload in self.layer_list}
        extractor = bdtorch.FeatureExtractor(
            self.encoder, layer_to_shape.keys(),
            layer_mapping=layer_map, detach=True)
        features = extractor.run(self.input_tensor)
        for layer, shape in layer_to_shape.items():
            self.assertEqual(features[layer].shape, shape)


class TestImageDataset(unittest.TestCase):
    ...


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFeatureExtractor)
    unittest.TextTestRunner(verbosity=2).run(suite)
