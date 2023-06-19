import argparse
import fnmatch
import glob
import json
import multiprocessing
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from argoverse.data_loading.argoverse_tracking_loader import \
    ArgoverseTrackingLoader
from argoverse.data_loading.synchronization_database import Nanosecond, Second
from argoverse.utils.camera_stats import RING_CAMERA_LIST
from argoverse.utils.metric_time import to_metric_time
from matplotlib import pyplot as plt
from PIL import Image


class Lidar2Depth:
    def __init__(
        self, input_log_dir: str, output_save_path: str, threshold: float = 0.90
    ) -> None:
        self.input_log_dir = input_log_dir
        self.output_save_path = output_save_path
        self.log_id = os.path.basename(input_log_dir)
        self.threshold = threshold

        # Load Argo data
        dataset = os.path.dirname(self.input_log_dir)
        self.argoverse_loader = ArgoverseTrackingLoader(dataset)
        self.argoverse_data = self.argoverse_loader.get(self.log_id)

        # Count the number of LiDAR ply files in the log dir
        self.lidar_frame_counter = len(
            glob.glob1(os.path.join(self.input_log_dir, "lidar"), "*.ply")
        )
        self.depth_data_dir_setup()
        self.intrinsics = {}

    def depth_data_dir_setup(self) -> None:
        if fnmatch.fnmatchcase(self.input_log_dir, "*" + "train" + "*"):
            self.save_name = os.path.join(self.output_save_path, "train")
            self.logid_type = "train"

        elif fnmatch.fnmatchcase(self.input_log_dir, "*" + "val" + "*"):
            self.save_name = os.path.join(self.output_save_path, "val")
            self.logid_type = "val"

        elif fnmatch.fnmatchcase(self.input_log_dir, "*" + "test" + "*"):
            self.save_name = os.path.join(self.output_save_path, "test")
            self.logid_type = "test"

        elif fnmatch.fnmatchcase(self.input_log_dir, "*" + "sample" + "*"):
            self.save_name = os.path.join(self.output_save_path, "sample")
            self.logid_type = "sample"

        for camera_name in RING_CAMERA_LIST:
            paths = [
                os.path.join(self.save_name, "depth", self.log_id, camera_name),
                os.path.join(self.save_name, "rgb", self.log_id, camera_name),
            ]
            for sub_path in paths:
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)

    def extract_lidar_image_pair(
        self, camera_ID: int, lidar_frame_idx: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        For the provided camera_ID and LiDAR ply file,
        extract rgb image and corresponding LiDAR points in the fov.
        """
        img = self.argoverse_data.get_image_sync(lidar_frame_idx, camera=camera_ID)
        self.calib = self.argoverse_data.get_calibration(camera_ID)
        pc = self.argoverse_data.get_lidar(lidar_frame_idx)
        uv = self.calib.project_ego_to_image(pc).T
        lidar_frame_idx_ = np.where(
            np.logical_and.reduce(
                (
                    uv[0, :] >= 0.0,
                    uv[0, :] < np.shape(img)[1] - 1.0,
                    uv[1, :] >= 0.0,
                    uv[1, :] < np.shape(img)[0] - 1.0,
                    uv[2, :] > 0,
                )
            )
        )
        lidar_image_projection_points = uv[:, lidar_frame_idx_]
        if lidar_image_projection_points is None:
            return np.array(img), None, None
        else:
            return np.array(img), lidar_image_projection_points, self.calib.K.tolist()

    def save_image_pair(
        self,
        camera_ID: int,
        img: np.ndarray,
        lidar_frame_idx: int,
        lidar_image_projection_points: np.ndarray,
        intrinsic,
    ) -> None:
        """
        Save the depth images and camera frame to the created dataset dir.
        """
        x_values = np.round(lidar_image_projection_points[0], 0).astype(int)
        y_values = np.round(lidar_image_projection_points[1], 0).astype(int)
        lidar_depth_val = lidar_image_projection_points[2]

        # Create a blank image to place lidar points as pixels with depth information
        sparse_depth_img = np.zeros(
            img.shape[:2], dtype=np.float32
        )  # keeping it float to maintain precision
        sparse_depth_img[y_values, x_values] = lidar_depth_val

        # Multiple to maintain precision, while model training, remember to divide by 256
        # NOTE: 0 denotes a null value, rather than actually zero depth in the saved depth map
        depth_rescaled = np.clip(sparse_depth_img, 0.0, 255.0) * 256.0
        depth_scaled = depth_rescaled.astype(np.uint16)
        depth_scaled = Image.fromarray(depth_scaled)
        raw_depth_path = os.path.join(
            self.save_name,
            "depth",
            self.log_id,
            str(camera_ID),
            str(lidar_frame_idx) + ".png",
        )
        depth_scaled.save(raw_depth_path)  # Save Depth image

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw_img_path = os.path.join(
            self.save_name,
            "rgb",
            self.log_id,
            str(camera_ID),
            str(lidar_frame_idx) + ".png",
        )
        cv2.imwrite(
            raw_img_path, img_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )  # Save RGB image
        self.intrinsics[
            raw_img_path.replace(self.save_name, "").lstrip("/")
        ] = intrinsic

    def frame2depth_mapping(self, camera_ID: int, lidar_frame_idx: str) -> None:
        """
        For your training dataloader, you will likely find it helpful to read image paths
        from a .txt file. We explicitly write to a .txt file all rgb image paths that have
        a corresponding sparse ground truth depth file along with focal length.
        """
        mapping_file = open(
            os.path.join(
                self.output_save_path,
                "argo_" + self.logid_type + self.log_id + "_files_with_gt.txt",
            ),
            "a",
        )
        depth_fn = os.path.join(
            "depth", str(self.log_id), camera_ID, str(lidar_frame_idx) + ".png"
        )
        rgb_fn = os.path.join(
            "rgb", str(self.log_id), camera_ID, str(lidar_frame_idx) + ".png"
        )
        mapping_file.write(f"{rgb_fn} {depth_fn}\n")

    def depth_extraction(self) -> None:
        """
        For every lidar file, extract ring camera frames and store it in the save dir
        along with depth map
        """
        distance = 2 if self.logid_type == "train" else 50
        time_step = 30
        cnt_images = 0
        old_pose = None
        old_timestamp = None
        for lidar_frame_idx in range(self.lidar_frame_counter):
            curr_pose = self.argoverse_data.get_pose(lidar_frame_idx).translation
            curr_timestamp = self.argoverse_data.lidar_timestamp_list[lidar_frame_idx]
            if lidar_frame_idx > 0:
                elapsed_time = to_metric_time(
                    ts=(curr_timestamp - old_timestamp), src=Nanosecond, dst=Second
                )
                pose_shift = (
                    sum(
                        [
                            (curr_coord - old_pose[i]) ** 2
                            for i, curr_coord in enumerate(curr_pose)
                        ]
                    )
                    ** 0.5
                )
                if (
                    pose_shift < distance and elapsed_time < time_step
                ):  # and compare_images(prev_img, img) > self.threshold :
                    continue
            old_pose = curr_pose
            old_timestamp = curr_timestamp
            for camera_ID in RING_CAMERA_LIST:
                # Extract camera frames and associated lidar points
                (
                    img,
                    lidar_image_projection_points,
                    calib,
                ) = self.extract_lidar_image_pair(camera_ID, lidar_frame_idx)
                if lidar_image_projection_points is None:
                    continue
                # Save image and depth map if LiDAR projection points exist
                # Save the above extracted images
                self.save_image_pair(
                    camera_ID,
                    img,
                    lidar_frame_idx,
                    lidar_image_projection_points,
                    calib,
                )
                cnt_images += 1

        return cnt_images


def worker(args):
    cnt = 0
    log_list, output_save_path = args
    for input_log_dir in log_list:
        cnt += Lidar2Depth(input_log_dir, output_save_path).depth_extraction()
    print(f"Local counter: {cnt}")
    return cnt


def main_worker(args):
    split = args.split
    base_path = args.base_path

    local_path_to_argoverse_splits = os.path.join(base_path, "argoverse-tracking")
    output_save_path = os.path.join(base_path, "argo_out")
    os.makedirs(output_save_path, exist_ok=True)
    folder = os.path.join(local_path_to_argoverse_splits, split)
    log_list = [f.path for f in os.scandir(folder) if f.is_dir()]
    log_list.sort()
    n_cpus = multiprocessing.cpu_count()

    chunk_s = len(log_list) // n_cpus + 1
    log_list = [
        log_list[i : min(i + chunk_s, len(log_list))]
        for i in range(0, len(log_list), chunk_s)
    ]
    print("Jobs: ", len(log_list))
    with multiprocessing.Pool(n_cpus) as p:
        res = p.imap_unordered(
            worker, zip(log_list, [output_save_path] * len(log_list))
        )
        print("TOT: ", sum(res))


if __name__ == "__main__":
    splits = {
        "train1": "https://s3.amazonaws.com/argoai-argoverse/tracking_train1_v1.1.tar.gz",
        "train2": "https://s3.amazonaws.com/argoai-argoverse/tracking_train2_v1.1.tar.gz",
        "train3": "https://s3.amazonaws.com/argoai-argoverse/tracking_train3_v1.1.tar.gz",
        "train4": "https://s3.amazonaws.com/argoai-argoverse/tracking_train4_v1.1.tar.gz",
        "val": "https://s3.amazonaws.com/argoai-argoverse/tracking_val_v1.1.tar.gz",
    }

    # Arguments
    parser = argparse.ArgumentParser(
        description="Argoverse Processing", conflict_handler="resolve"
    )
    parser.add_argument("--split", type=str, required=True, choices=list(splits.keys()))
    parser.add_argument("--base-path", type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(
        os.path.join(args.base_path, splits[args.split].split("/")[-1])
    ):
        os.system(f"wget '{splits[args.split]}' -P {args.base_path}")
    os.system(
        f"tar -xzf {os.path.join(args.base_path, splits[args.split].split('/')[-1])} -C {args.base_path}"
    )
    main_worker(args)
