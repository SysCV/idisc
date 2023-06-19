import argparse
import json
import multiprocessing
import os
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from dgp.datasets import SynchronizedSceneDataset
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
RING_CAMERAS = ["CAMERA_01", "CAMERA_05", "CAMERA_06", "CAMERA_09"]


def sample_ddad(dataset: SynchronizedSceneDataset, idx):
    assert (
        dataset.dataset_item_index is not None
    ), "Index is not built, select datums before getting elements."
    # Get dataset item index
    scene_idx, sample_idx_in_scene, datum_names = dataset.dataset_item_index[idx]

    # All sensor data (including pose, point clouds and 3D annotations are
    # defined with respect to the sensor's reference frame captured at that
    # corresponding timestamp. In order to move to a locally consistent
    # reference frame, you will need to use the "pose" that specifies the
    # ego-pose of the sensor with respect to the local (L) frame (pose_LS).

    datums_with_context = dict()
    filenames = []
    for datum_name in datum_names:

        acc_back, acc_forward = 0, 0
        if dataset.accumulation_context:
            accumulation_context = dataset.accumulation_context.get(
                datum_name.lower(), (0, 0)
            )
            acc_back, acc_forward = accumulation_context

        # We need to fetch our datum's data for all time steps we care about. If our datum's index is i,
        # then to support the requested backward/forward context, we need indexes i- backward ... i + forward.
        # However if we also have accumulation, our first sample index = (i - backward) also needs data starting from
        # index = i - backward - acc_back (it accumulates over a window of size (acc_back, acc_forward).
        # The final sample will need i + forward + acc_forward. Combined, we need samples from
        # [i-backward_context - acc_back, i+forward_context + acc_forward].
        datum_list = [
            dataset.get_datum_data(scene_idx, sample_idx_in_scene + offset, datum_name)
            for offset in range(
                -1 * (dataset.backward_context + acc_back),
                dataset.forward_context + acc_forward + 1,
            )
        ]

        # if acc_back != 0 or acc_forward != 0:
        #   # Make sure we have the right datum type
        #   assert 'point_cloud' in datum_list[0], "Accumulation is only defined for radar and lidar currently."
        #   # Our datum list now has samples ranging from [i-backward_context-acc_back to i+forward_context+acc_forward]
        #   # We instead need a list that ranges from [i-backward_context to i+forward_context] AFTER accumulation
        #   # This means the central sample in our datum list starts at index = acc_back.
        #   datum_list = [
        #       accumulate_points(
        #           datum_list[k - acc_back:k + acc_forward + 1], datum_list[k],
        #           dataset.transform_accumulated_box_points
        #       ) for k in range(acc_back,
        #                         len(datum_list) - acc_forward)
        #   ]

        datums_with_context[datum_name] = datum_list
        scene_dir = dataset.scenes[scene_idx].directory
        datum = dataset.get_datum(scene_idx, sample_idx_in_scene, datum_name)
        filenames.append(os.path.join(scene_dir, datum.datum.image.filename))

    # We now have a dictionary of lists, swap the order to build context windows
    context_window = []
    for t in range(dataset.backward_context + dataset.forward_context + 1):
        context_window.append(
            [datums_with_context[datum_name][t] for datum_name in datum_names]
        )

    return context_window, scene_idx, sample_idx_in_scene, filenames


def worker(args):
    idxs, output_save_path, kind, dataset = args
    cnt = 0
    old_pose, old_timestamp = None, None
    min_distance = 2 if kind == "train" else 50
    time_step = 20
    for i, idx in tqdm(enumerate(idxs), total=len(idxs)):
        cameras, scene_idx, sample_idx, filenames = sample_ddad(dataset, idx)

        cameras = cameras[0]
        curr_pose = cameras[0]["pose"].tvec
        curr_timestamp = cameras[0]["timestamp"]
        if i > 0:
            elapsed_time = (curr_timestamp - old_timestamp) / 1e6
            pose_shift = (
                sum(
                    [
                        (curr_coord - old_pose[j]) ** 2
                        for j, curr_coord in enumerate(curr_pose)
                    ]
                )
                ** 0.5
            )
            if pose_shift < min_distance and elapsed_time < time_step:
                continue

        old_pose = curr_pose
        old_timestamp = curr_timestamp

        for j, camera in enumerate(cameras):
            if camera["datum_name"] == "LIDAR":
                continue
            rgb = camera["rgb"]  # PIL.Image
            # (H,W) numpy.ndarray, generated from 'lidar'
            depth = (256 * np.clip(camera["depth"], 0.0, 255.0)).astype(np.uint16)
            scene_code, _, camera_name, filename = filenames[j].split("/")[-4:]
            rgb_dir = os.path.join(
                output_save_path, str(scene_code), "rgb", camera_name
            )
            rgb_fn = os.path.join(rgb_dir, filename)
            depth_dir = os.path.join(
                output_save_path, str(scene_code), "depth", camera_name
            )
            depth_fn = os.path.join(depth_dir, filename)

            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            rgb.save(rgb_fn)
            Image.fromarray(depth).save(depth_fn)
            cnt += 1

    return cnt


def main_worker(args):
    split = args.split
    base_path = args.base_path
    local_path_to_argoverse_splits = os.path.join(
        base_path, "ddad_train_val", "ddad.json"
    )
    output_save_path = os.path.join(base_path, "ddad_results")
    n_cpus = multiprocessing.cpu_count()
    os.makedirs(output_save_path, exist_ok=True)

    dataset = SynchronizedSceneDataset(
        local_path_to_argoverse_splits,
        datum_names=["lidar", *RING_CAMERAS],
        generate_depth_from_datum="lidar",
        split=split,
    )

    chunk_s = len(dataset) // n_cpus + 1
    idx_list = [
        list(range(len(dataset)))[i : min(i + chunk_s, len(dataset))]
        for i in range(0, len(dataset), chunk_s)
    ]

    with multiprocessing.Pool(n_cpus) as p:
        res = p.imap_unordered(
            worker,
            zip(
                idx_list,
                [output_save_path] * len(idx_list),
                [split] * len(idx_list),
                [deepcopy(dataset) for _ in range(len(idx_list))],
            ),
        )
        print("Total samples:", sum(res))


if __name__ == "__main__":
    # Arguments
    LINK_DDAD = "https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar"
    parser = argparse.ArgumentParser(
        description="Process DDAD", conflict_handler="resolve"
    )
    parser.add_argument("--split", type=str, required=True, choices=["train", "val"])
    parser.add_argument("--base-path", type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.base_path, LINK_DDAD.split("/")[-1])):
        os.system(f"wget '{LINK_DDAD}' -P {args.base_path}")
    os.system(
        f"tar -xzf {os.path.join(args.base_path, LINK_DDAD.split('/')[-1])} -C {args.base_path}"
    )
    main_worker(args)
