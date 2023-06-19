"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    min_depth = 0.01
    max_depth = 1000
    train_split = ""
    test_split = ""

    def __init__(self, test_mode, base_path, benchmark, normalize) -> None:
        super().__init__()
        self.base_path = base_path
        self.split_file = self.train_split if not test_mode else self.test_split
        self.test_mode = test_mode
        self.normalization_stats = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
        self.benchmark = benchmark
        self.dataset = []
        if normalize:
            self.normalization_stats = {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }

    def load_dataset(self):
        raise NotImplementedError

    def __len__(self):
        """Total number of samples of data."""
        return len(self.dataset)

    def _augmentation_space(
        self, height, width
    ) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return (
            {
                "Identity": {
                    "function": lambda x: x,
                    "kwargs": dict(),
                    "geometrical": False,
                    "weight": 1,
                },
                "Brightness": {
                    "function": TF.adjust_brightness,
                    "kwargs": dict(
                        brightness_factor=10
                        ** np.random.uniform(
                            -self.random_brightness, self.random_brightness
                        )
                    ),
                    "geometrical": False,
                },
                "Contrast": {
                    "function": TF.adjust_contrast,
                    "kwargs": dict(
                        contrast_factor=10
                        ** np.random.uniform(
                            -self.random_contrast, self.random_contrast
                        )
                    ),
                    "geometrical": False,
                },
                "Saturation": {
                    "function": TF.adjust_saturation,
                    "kwargs": dict(
                        saturation_factor=10
                        ** np.random.uniform(
                            -self.random_saturation, self.random_saturation
                        )
                    ),
                    "geometrical": False,
                },
                "Gamma": {
                    "function": TF.adjust_gamma,
                    "kwargs": dict(
                        gamma=np.random.uniform(
                            1.0 - self.random_gamma, 1.0 + self.random_gamma
                        )
                    ),
                    "geometrical": False,
                },
                "Hue": {
                    "function": TF.adjust_hue,
                    "kwargs": dict(
                        hue_factor=np.random.uniform(-self.random_hue, self.random_hue)
                    ),
                    "geometrical": False,
                },
                "Sharpness": {
                    "function": TF.adjust_sharpness,
                    "kwargs": dict(
                        sharpness_factor=10
                        ** np.random.uniform(
                            -self.random_sharpness, self.random_sharpness
                        )
                    ),
                    "geometrical": False,
                },
                "Posterize": {
                    "function": TF.posterize,
                    "kwargs": dict(
                        bits=8 - np.random.randint(0, self.random_posterize)
                    ),
                    "geometrical": False,
                },
                "Solarize": {
                    "function": TF.solarize,
                    "kwargs": dict(
                        threshold=int(
                            255 * (1 - np.random.uniform(0, self.random_solarize))
                        )
                    ),
                    "geometrical": False,
                },
                "Equalize": {
                    "function": TF.equalize,
                    "kwargs": dict(),
                    "geometrical": False,
                },
                "Autocontrast": {
                    "function": TF.autocontrast,
                    "kwargs": dict(),
                    "geometrical": False,
                },
                "Translation": {
                    "function": TF.affine,
                    "kwargs": dict(
                        angle=0,
                        scale=1,
                        translate=[
                            int(
                                width
                                * np.random.uniform(
                                    -self.random_translation, self.random_translation
                                )
                            ),
                            int(
                                height
                                * np.random.uniform(
                                    -self.random_translation, self.random_translation
                                )
                            ),
                        ],
                        shear=0,
                    ),
                    "geometrical": True,
                },
                "Scale": {
                    "function": TF.affine,
                    "kwargs": dict(
                        angle=0,
                        scale=np.random.uniform(1.0, 1.0 + self.random_scale),
                        translate=[0, 0],
                        shear=0,
                    ),
                    "geometrical": True,
                },
                "Rotation": {
                    "function": TF.affine,
                    "kwargs": dict(
                        angle=np.random.uniform(
                            -self.random_rotation, self.random_rotation
                        ),
                        scale=1,
                        translate=[0, 0],
                        shear=0,
                    ),
                    "geometrical": True,
                },
            },
            {
                "Identity": 1,
                "Brightness": self.brightness_p,
                "Contrast": self.contrast_p,
                "Saturation": self.saturation_p,
                "Gamma": self.gamma_p,
                "Hue": self.hue_p,
                "Sharpness": self.sharpness_p,
                "Posterize": self.posterize_p,
                "Solarize": self.solarize_p,
                "Equalize": self.equalize_p,
                "Autocontrast": self.autocontrast_p,
                "Translation": self.translation_p,
                "Scale": self.scale_p,
                "Rotation": self.rotation_p,
            },
        )

    def preprocess_crop(self, image, gts, info):
        raise NotImplementedError

    def transform_train(self, image, gts, info=None):
        width, height = image.size
        # Horizontal flip
        if np.random.uniform(0.0, 1.0) < 0.5:
            image = TF.hflip(image)
            for k, v in gts.items():
                gts[k] = TF.hflip(v)
                if "normal_gt" in k:
                    v = np.array(gts[k])
                    v[..., 0] = 255 - v[..., 0]
                    gts[k] = Image.fromarray(v)
            info["camera_intrinsics"][0, 2] = width - info["camera_intrinsics"][0, 2]

        augmentations_dict, augmentations_weights = self._augmentation_space(
            height, width
        )
        augmentations_probs = np.array(list(augmentations_weights.values()))

        num_augmentations = np.random.choice(
            list(range(1, 5)), size=1, p=[0.3, 0.4, 0.2, 0.1]
        )

        current_ops = np.random.choice(
            list(augmentations_weights.keys()),
            size=num_augmentations,
            replace=False,
            p=augmentations_probs / np.sum(augmentations_probs),
        )
        for op_name in current_ops:
            op_meta = augmentations_dict[op_name]
            if op_meta["geometrical"]:
                image = op_meta["function"](
                    image,
                    interpolation=TF.InterpolationMode.BICUBIC,
                    **op_meta["kwargs"]
                )
                for k, v in gts.items():
                    gts[k] = op_meta["function"](
                        v,
                        interpolation=TF.InterpolationMode.NEAREST,
                        **op_meta["kwargs"]
                    )
                info["camera_intrinsics"][0, 0] = (
                    info["camera_intrinsics"][0, 0] * op_meta["kwargs"]["scale"]
                )
                info["camera_intrinsics"][1, 1] = (
                    info["camera_intrinsics"][1, 1] * op_meta["kwargs"]["scale"]
                )
                info["camera_intrinsics"][0, 2] = (
                    info["camera_intrinsics"][0, 2] * op_meta["kwargs"]["scale"]
                    - op_meta["kwargs"]["translate"][0] * width
                )
                info["camera_intrinsics"][1, 2] = (
                    info["camera_intrinsics"][1, 2] * op_meta["kwargs"]["scale"]
                    - op_meta["kwargs"]["translate"][1] * height
                )
            else:
                image = op_meta["function"](image, **op_meta["kwargs"])

        return image, gts, info

    def transform(self, image, gts=None, info=None):
        image, gts, info = self.preprocess_crop(image, gts, info)
        image = Image.fromarray(image)
        for k, v in gts.items():
            gts[k] = Image.fromarray(v)
        if not self.test_mode:
            image, gts, info = self.transform_train(image, gts, info)

        image = TF.normalize(TF.to_tensor(image), **self.normalization_stats)

        for k, v in gts.items():
            v = np.array(v)
            if "gt" in k and v.shape[-1] > 1 and v.ndim > 2:
                v = v / 127.5 - 1
            gts[k] = TF.to_tensor(v)

        return image, gts, info

    def eval_mask(self, valid_mask):
        return valid_mask
