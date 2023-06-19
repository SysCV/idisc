import os

import cv2
import numpy as np
import torch
from PIL import Image

from .dataset import BaseDataset


class DiodeDataset(BaseDataset):
    CAM_INTRINSIC = {
        "ALL": torch.tensor([[886.81, 0, 512], [0, 927.06, 384], [0, 0, 1]])
    }
    min_depth = 0.01
    max_depth = 10
    test_split = "diode_indoor_val.txt"
    train_split = "diode_indoor_train.txt"

    def __init__(
        self,
        test_mode,
        base_path,
        depth_scale=256,
        crop=None,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.crop = crop
        self.height = 768
        self.width = 1024
        self.rescale = 1.6
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
                    if depth_map == "None":
                        self.invalid_depth_num += 1
                        continue
                    img_info["annotation_filename"] = os.path.join(
                        self.base_path, depth_map
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
        depth = (
            np.asarray(
                Image.open(
                    os.path.join(
                        self.base_path, self.dataset[idx]["annotation_filename"]
                    )
                )
            ).astype(np.float32)
            / self.depth_scale
        )
        info = self.dataset[idx].copy()
        info["camera_intrinsics"] = self.CAM_INTRINSIC["ALL"].clone()

        image, gts, info = self.transform(
            image=image,
            gts={"depth": depth},
            info=info,
        )
        return {"image": image, "gts": gts, "info": info}

    # def get_pointcloud_mask(self, shape):
    #     mask = np.zeros(shape)
    #     height_start, height_end = 45, self.height - 9
    #     width_start, width_end = 41, self.width - 39
    #     mask[height_start:height_end, width_start:width_end] = 1
    #     return mask

    def preprocess_crop(self, image, gts=None, info=None):
        height_start, width_start = 0, 0
        height_end, width_end = height_start + self.height, width_start + self.width
        # image = image[height_start:height_end, width_start:width_end]
        image = cv2.resize(
            image,
            (int(image.shape[1] / self.rescale), int(image.shape[0] / self.rescale)),
            interpolation=cv2.INTER_LINEAR,
        )
        info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] - width_start
        info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] - height_start
        info["camera_intrinsics"][0, 0] = info["camera_intrinsics"][0, 0] / self.rescale
        info["camera_intrinsics"][1, 1] = info["camera_intrinsics"][1, 1] / self.rescale
        info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] / self.rescale
        info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] / self.rescale
        new_gts = {}
        if "depth" in gts:
            # depth = depth[height_start:height_end, width_start:width_end]
            depth = cv2.resize(
                gts["depth"],
                (
                    int(gts["depth"].shape[1] / self.rescale),
                    int(gts["depth"].shape[0] / self.rescale),
                ),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = depth > self.min_depth
            mask = self.eval_mask(mask).astype(np.uint8)
            new_gts["depth_gt"], new_gts["depth_mask"] = depth, mask
        return image, new_gts, info

    def eval_mask(self, valid_mask):
        """Do grag_crop or eigen_crop for testing"""
        # return valid_mask
        border_mask = np.zeros_like(valid_mask)
        border_mask[45:471, 41:601] = 1
        return np.logical_and(valid_mask, border_mask)
