import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

from .dataset import BaseDataset


class ArgoverseDataset(BaseDataset):
    min_depth = 0.01
    max_depth = 150.0
    test_split = "argo_val.txt"
    train_split = "argo_train.txt"
    intrisics_file = "argo_intrinsics.json"

    def __init__(
        self,
        test_mode,
        base_path,
        depth_scale=256,
        crop=None,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        rescale=1.5,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.crop = crop
        self.height = 870
        self.width = 1920
        self.height_start, self.width_start = 180, 0
        self.height_end, self.width_end = (
            self.height_start + self.height,
            self.width_start + self.width,
        )

        self.rescale = rescale
        self.kernel = np.ones((3, 3), np.float32)
        # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        with open(os.path.join(self.base_path, self.intrisics_file)) as f:
            intrisics = json.load(f)
        with open(os.path.join(self.base_path, self.split_file)) as f:
            for line in f:
                img_info = dict()
                if not self.benchmark:  # benchmark test
                    depth_map = line.strip().split(" ")[1]
                    img_info["annotation_filename"] = os.path.join(
                        self.base_path, depth_map
                    )
                img_name = line.strip().split(" ")[0]
                img_info["image_filename"] = os.path.join(self.base_path, img_name)
                img_info["camera_intrinsics"] = torch.tensor(
                    intrisics[img_name]
                ).squeeze()[:, :3]
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
        image = np.asarray(Image.open(self.dataset[idx]["image_filename"]))
        depth = (
            np.asarray(Image.open(self.dataset[idx]["annotation_filename"])).astype(
                np.float32
            )
            / self.depth_scale
        )
        info = self.dataset[idx].copy()
        info["camera_intrinsics"] = self.dataset[idx]["camera_intrinsics"].clone()
        image, gts, info = self.transform(
            image=image,
            gts={"depth": depth},
            info=info,
        )
        return {"image": image, "gt": gts["gt"], "mask": gts["mask"]}

    def preprocess_crop(self, image, gts=None, info=None):
        image = image[
            self.height_start : self.height_end, self.width_start : self.width_end
        ]
        image = cv2.resize(
            image,
            (int(image.shape[1] / self.rescale), int(image.shape[0] / self.rescale)),
            interpolation=cv2.INTER_LINEAR,
        )
        info["camera_intrinsics"][0, 2] = (
            info["camera_intrinsics"][0, 2] - self.width_start
        )
        info["camera_intrinsics"][1, 2] = (
            info["camera_intrinsics"][1, 2] - self.height_start
        )
        info["camera_intrinsics"][0, 0] = info["camera_intrinsics"][0, 0] / self.rescale
        info["camera_intrinsics"][1, 1] = info["camera_intrinsics"][1, 1] / self.rescale
        info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] / self.rescale
        info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] / self.rescale
        new_gts = {}
        if "depth" in gts:
            depth = gts["depth"][
                self.height_start : self.height_end, self.width_start : self.width_end
            ]
            mask = depth > self.min_depth
            if self.test_mode:
                mask = np.logical_and(mask, depth < self.max_depth)
            mask = mask.astype(np.uint8)
            new_gts["gt"], new_gts["mask"] = depth, mask
        return image, new_gts, info
