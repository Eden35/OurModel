"""Numpy dataset and paired augmentations for the paper training recipe."""
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NpySegmentationDataset(Dataset):
    def __init__(self, root, split_folder="folder1", split="train", output_size=None, augment=False):
        self.root = Path(root)
        self.output_size = tuple(output_size) if output_size else None
        self.augment = bool(augment and split == "train")
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split: {}".format(split))
        # Existing split folders resolve relative to the repository, independent of cwd.
        folder = Path(split_folder)
        if not folder.is_absolute():
            folder = Path(__file__).resolve().parent / folder
        list_path = folder / (folder.name + "_" + split + ".list")
        self.names = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = np.load(str(self.root / "image" / name), allow_pickle=False)
        label = np.load(str(self.root / "label" / (Path(name).stem + ".npy")), allow_pickle=False)
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        label = np.squeeze(label)
        if image.ndim != 3 or image.shape[-1] != 3 or label.ndim != 2 or image.shape[:2] != label.shape:
            raise ValueError("Expected aligned RGB/grayscale image and 2D label: {}".format(name))
        if not np.isfinite(image).all() or not np.isfinite(label).all():
            raise ValueError("Non-finite image or label: {}".format(name))
        labels = np.unique(label)
        if not (np.isin(labels, [0, 1]).all() or np.isin(labels, [0, 255]).all()):
            raise ValueError("Binary labels must use 0/1 or 0/255: {}".format(name))
        is_uint8 = image.dtype == np.uint8
        # uint8 is always scaled, even if a dark image contains only 0 and 1.
        image = torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1)
        if is_uint8:
            image = image / 255.0
        elif image.min() < 0 or image.max() > 1:
            raise ValueError("Float image values must be in [0,1]: {}".format(name))
        label = torch.from_numpy((label > 0).astype(np.float32)).unsqueeze(0)
        if self.output_size:
            image = F.interpolate(image[None], size=self.output_size, mode="bilinear", align_corners=False)[0]
            label = F.interpolate(label[None], size=self.output_size, mode="nearest")[0]
        if self.augment:
            if random.random() < 0.5:
                image, label = image.flip(-1), label.flip(-1)
            if random.random() < 0.5:
                image, label = image.flip(-2), label.flip(-2)
            scale = random.uniform(0.8, 1.2)
            angle = math.radians(random.uniform(-30.0, 30.0))
            height, width = image.shape[-2:]
            # Inverse transform in normalized coordinates: rotation is defined
            # in pixel space and scaling actually zooms the image in/out.
            theta = image.new_tensor([
                [math.cos(angle) / scale, -math.sin(angle) * height / width / scale, 0.0],
                [math.sin(angle) * width / height / scale, math.cos(angle) / scale, 0.0],
            ])
            grid = F.affine_grid(theta[None], image[None].size(), align_corners=False)
            image = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
            label = F.grid_sample(label[None], grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
            zero = image.abs().sum(dim=0, keepdim=True) == 0
            image = image + zero.float() * torch.randn_like(image) * 0.05
        image = (image - 0.5) / 0.5
        return image, label[0].long(), name
