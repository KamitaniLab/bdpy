import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from bdpy.dl.torch.dataset import ImageDataset

# Use non-square images (H=4, W=6, C=3) to fully discriminate every axis.
# A square image (H=W) cannot distinguish (3, H, H) from (H, H, 3).
_H, _W = 4, 6


def _save_image(path: Path, r: int, g: int, b: int, h: int = _H, w: int = _W) -> None:
    data = np.zeros((h, w, 3), dtype=np.uint8)
    data[:, :, 0] = r
    data[:, :, 1] = g
    data[:, :, 2] = b
    Image.fromarray(data).save(path)


class TestImageDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        _save_image(root / "a.jpg", r=200, g=100, b=50)
        _save_image(root / "b.jpg", r=10, g=20, b=30)
        _save_image(root / "c.jpg", r=0, g=128, b=255)
        self.root = root

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_getitem_returns_chw_shape(self):
        dataset = ImageDataset(self.root, stimulus_names=["a"])
        arr, _ = dataset[0]
        self.assertEqual(arr.shape, (3, _H, _W))

    def test_getitem_preserves_channels(self):
        # R=200 G=100 B=50 — verifies C axis maps to the correct channel.
        dataset = ImageDataset(self.root, stimulus_names=["a"])
        arr, _ = dataset[0]
        self.assertTrue(np.allclose(arr[0], 200 / 255.0))
        self.assertTrue(np.allclose(arr[1], 100 / 255.0))
        self.assertTrue(np.allclose(arr[2], 50 / 255.0))

    def test_dataloader_integration_batch_shape(self):
        dataset = ImageDataset(self.root, stimulus_names=["a", "b"])
        loader = DataLoader(dataset, batch_size=2)
        batch_images, _ = next(iter(loader))
        self.assertEqual(tuple(batch_images.shape), (2, 3, _H, _W))

    def test_value_range_normalized_to_unit_interval(self):
        dataset = ImageDataset(self.root, stimulus_names=["a", "b", "c"])
        for i in range(len(dataset)):
            arr, _ = dataset[i]
            self.assertGreaterEqual(float(arr.min()), 0.0)
            self.assertLessEqual(float(arr.max()), 1.0)

    def test_len_matches_stimulus_names(self):
        dataset = ImageDataset(self.root, stimulus_names=["a", "b", "c"])
        self.assertEqual(len(dataset), 3)

    def test_explicit_stimulus_names_respected(self):
        dataset = ImageDataset(self.root, stimulus_names=["a", "c"])
        self.assertEqual(len(dataset), 2)
        _, label0 = dataset[0]
        _, label1 = dataset[1]
        self.assertEqual(label0, "a")
        self.assertEqual(label1, "c")

    def test_auto_detected_stimulus_names_use_stem(self):
        dataset = ImageDataset(self.root)
        self.assertEqual(set(dataset._stimulus_names), {"a", "b", "c"})

    def test_explicit_stimulus_names_preserve_input_order(self):
        dataset = ImageDataset(self.root, stimulus_names=["c", "a", "b"])
        labels = [dataset[i][1] for i in range(len(dataset))]
        self.assertEqual(labels, ["c", "a", "b"])

    def test_auto_detected_stimulus_names_are_sorted(self):
        dataset = ImageDataset(self.root)
        self.assertEqual(dataset._stimulus_names, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
