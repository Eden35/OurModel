"""Verify binary npy input and paired geometric augmentation semantics."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from Datasets.evidence_guided import NpySegmentationDataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ('image', 'label', 'split'):
            (self.root / name).mkdir()
        (self.root / 'split/split_train.list').write_text('a.npy\n')

    def tearDown(self):
        self.temp.cleanup()

    def save(self, image, mask):
        np.save(str(self.root / 'image/a.npy'), image)
        np.save(str(self.root / 'label/a.npy'), mask)

    def test_uint8_scale_and_mask_encodings(self):
        image = np.ones((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        self.save(image, mask)
        dataset = NpySegmentationDataset(self.root, str(self.root / 'split'), 'train')
        x, y, name = dataset[0]
        self.assertTrue(torch.allclose(x, torch.full_like(x, 2 / 255 - 1)))
        self.assertEqual(y.sum().item(), 256)
        self.assertEqual(name, 'a.npy')
        self.save(image, mask // 255)
        self.assertTrue(torch.equal(dataset[0][1], y))

    def test_scale_actually_changes_region_size_and_remains_paired(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1
        self.save(np.repeat(mask[..., None].astype(np.float32), 3, axis=2), mask)
        dataset = NpySegmentationDataset(self.root, str(self.root / 'split'), 'train', augment=True)
        areas = []
        for scale in (.8, 1.2):
            with patch('Datasets.evidence_guided.random.uniform', side_effect=[scale, 0.]), patch('Datasets.evidence_guided.random.random', return_value=1.):
                image, label, _ = dataset[0]
            # Bilinear edges and nearest mask agree under the same transform.
            self.assertTrue(torch.equal(image[0] > 0, label.bool()))
            areas.append(label.sum().item())
        self.assertLess(areas[0], 32 * 32)
        self.assertGreater(areas[1], 32 * 32)

    def test_invalid_nonbinary_mask_rejected(self):
        self.save(np.zeros((16, 16), dtype=np.uint8), np.full((16, 16), 17, dtype=np.uint8))
        dataset = NpySegmentationDataset(self.root, str(self.root / 'split'))
        with self.assertRaisesRegex(ValueError, 'Binary labels'):
            dataset[0]


if __name__ == '__main__':
    unittest.main()
