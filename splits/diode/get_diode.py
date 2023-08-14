from collections import defaultdict
from copy import deepcopy
import os
import numpy as np
import multiprocessing
from PIL import Image


def worker(args):
    keys_list, fns, save_dir, src_dir = args
    fp = open(os.path.join(src_dir, f"diode_{keys_list[0].split('/')[-1]}-{keys_list[-1].split('/')[-1]}.txt"), "w")
    cnt = 0
    for dirpath in keys_list:
        for i, f in enumerate(fns[dirpath]):
            dest_dir = os.path.join(save_dir, dirpath)
            f = f.split(".")[0]

            depth = np.load(os.path.join(src_dir, dirpath, f+"_depth.npy")).astype(np.float32)
            depth_mask = np.load(os.path.join(src_dir, dirpath, f+"_depth_mask.npy"))
            depth[depth_mask < 1e-6] = 0
            depth = np.clip(depth, 0.0, 100)

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)
            depth_dst = os.path.join(dest_dir, f+"_depth.png")
            img_dst = os.path.join(dest_dir, f+".png")
            img_src = os.path.join(src_dir, dirpath, f+".png")
            
            Image.fromarray((256.0 * depth[..., 0]).astype(np.uint16)).save(depth_dst)
            os.system(f"cp {img_src} {img_dst}")
            fp.write(f"{img_dst.split(save_dir)[-1].strip('/')} {depth_dst.split(save_dir)[-1].strip('/')}\n")
            cnt += 1
    fp.close()
    return cnt


def main_worker(kind):
    local_path_to_splits = (os.environ["TMPDIR"])
    output_save_path = (os.environ["TMPDIR"]+"/diode")
    os.makedirs(output_save_path, exist_ok=True)
    folder = os.path.join(local_path_to_splits, kind)
    n_cpus = multiprocessing.cpu_count()

    fns = defaultdict(list)
    for (dirpath, dirnames, filenames) in os.walk(folder):
        dirpath = dirpath.split(local_path_to_splits)[-1].strip("/")
        if filenames:
            fns[dirpath].extend(filenames)

    for dirpath, filenames in fns.items():
        for i, f in enumerate(filenames):
            if "txt" in f:
                fns[dirpath].remove(f)
                continue
            fns[dirpath][i] = f.split(".")[0].replace("_depth", "").replace("_mask", "")

        fns[dirpath] = np.unique(fns[dirpath])

    chunk_s = len(fns) // n_cpus + 1
    keys_list = list(fns.keys())
    keys_list = [keys_list[i:min(i+chunk_s, len(keys_list))] for i in range(0, len(keys_list), chunk_s)]

    with multiprocessing.Pool(n_cpus) as p:
        res = p.imap_unordered(worker, zip(keys_list, [deepcopy(fns)] * len(keys_list), [output_save_path] * len(keys_list), [local_path_to_splits] * len(keys_list)))
        print("TOT: ", sum(res))

    # merge the txt files into the final ones
    fp_all = open(os.path.join(output_save_path, f"diode_{kind}.txt"), "w")
    fp_indoor = open(os.path.join(output_save_path, f"diode_indoor_{kind}.txt"), "w")
    fp_outdoor = open(os.path.join(output_save_path, f"diode_outdoor_{kind}.txt"), "w")
    for text in os.listdir(local_path_to_splits):
        if "txt" not in text:
            continue
        with open(os.path.join(local_path_to_splits, text)) as f:
            for line in f:
                fp_all.write(line)
                if "indoor" in line:
                    fp_indoor.write(line)
                if "outdoor" in line:
                    fp_outdoor.write(line)
        os.remove(os.path.join(local_path_to_splits, text))

    fp_all.close()
    fp_indoor.close()
    fp_outdoor.close()
    

if __name__ == '__main__':
    temp_dir = os.environ.get("TMPDIR", os.environ["HOME"])
    os.environ["TMPDIR"] = temp_dir
    for kind in ["val", "train"]:
        if not os.path.exists(os.path.join(temp_dir, kind+'.tar.gz')):
            os.system(f"wget 'http://diode-dataset.s3.amazonaws.com/{kind}.tar.gz' -P {temp_dir}")
            os.system(f"tar -xzf {os.path.join(temp_dir, kind+'.tar.gz')} -C {temp_dir}")
        main_worker(kind)
        # if save space
        # os.remove(os.path.join(temp_dir, kind+'.tar.gz'))
