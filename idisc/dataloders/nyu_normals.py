"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from .dataset import BaseDataset


class NYUNormalsDataset(BaseDataset):
    CAM_INTRINSIC = {
        "ALL": torch.tensor(
            [
                [5.1885790117450188e02, 0, 3.2558244941119034e02],
                [0, 5.1946961112127485e02, 2.5373616633400465e02],
                [0, 0, 1],
            ]
        )
    }
    min_depth = 0.01
    max_depth = 10
    test_split = "nyu_test.txt"
    train_split = "nyu_train.txt"

    def __init__(
        self,
        test_mode,
        base_path,
        depth_scale=1000,
        crop=None,
        benchmark=False,
        augmentations_db={},
        masked=True,
        normalize=True,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.crop = crop
        self.height = 480
        self.width = 640
        self.masked = masked

        # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        with open(os.path.join(self.base_path, self.split_file)) as f:
            for line in f:
                img_info = dict()
                if not self.benchmark:  # benchmark test
                    depth_map = line.strip().split(" ")[1]
                    normals_map = line.strip().split(" ")[2]
                    img_info["annotation_filename_depth"] = os.path.join(
                        self.base_path, depth_map
                    )
                    img_info["annotation_filename_normals"] = os.path.join(
                        self.base_path, normals_map
                    )
                img_name = line.strip().split(" ")[0]
                img_info["image_filename"] = os.path.join(self.base_path, img_name)

                self.dataset.append(img_info)
        print(
            f"Loaded {len(self.dataset)} images. Totally {self.invalid_depth_num} invalid pairs are filtered"
        )

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        image = np.asarray(
            Image.open(
                os.path.join(self.base_path, self.dataset[idx]["image_filename"])
            )
        )
        # depth = np.asarray(Image.open(os.path.join(self.base_path, self.dataset[idx]["annotation_filename_depth"]))).astype(np.float32) / self.depth_scale
        normals = np.asarray(
            Image.open(
                os.path.join(
                    self.base_path, self.dataset[idx]["annotation_filename_normals"]
                )
            )
        ).astype(np.uint8)[..., :3]
        info = self.dataset[idx].copy()
        info["camera_intrinsics"] = self.CAM_INTRINSIC["ALL"].clone()
        image, gts, info = self.transform(
            image=image,
            gts={"normals": normals},
            info=info,
        )
        return {"image": image, "gt": gts["gt"], "mask": gts["mask"]}

    def get_pointcloud_mask(self, shape):
        mask = np.zeros(shape)
        height_start, height_end = 45, self.height - 9
        width_start, width_end = 41, self.width - 39
        mask[height_start:height_end, width_start:width_end] = 1
        return mask

    def preprocess_crop(self, image, gts=None, info=None):
        height_start, height_end = 0, self.height
        width_start, width_end = 0, self.width
        image = image[height_start:height_end, width_start:width_end]
        new_gts = {}
        if "normals" in gts:
            normals = gts["normals"]
            mask = (normals.sum(axis=-1) > 0).astype(np.uint8)
            new_gts["gt"] = normals
            new_gts["mask"] = mask

        info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] - width_start
        info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] - height_start
        return image, new_gts, info

    def eval_mask(self, valid_mask):
        """Do grag_crop or eigen_crop for testing"""

        border_mask = np.zeros_like(valid_mask)
        border_mask[45:471, 41:601] = 1
        return np.logical_and(valid_mask, border_mask)
