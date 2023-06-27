## Prepare datasets

Warninig: the code works for tarballs with both data and splits in it. For instance, NYU comes as a zip file, we suggest to re-comprress it, with split files, as tarball.

Before training, the data need to be unzipped in the ``<BASE-PATH>/datasets``
If your folder structure is different, you may need to change the corresponding paths in the config files.
You can look into the ``../splits`` folder for the pre-processed splits lists.

### **KITTI**

Download the official dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction), including the raw data (about 200G) and fine-grained ground-truth depth maps. 
Unzip files and copy the [split files](../splits/kitti) in the kitti folder. The files not contained in the splits can be deleted to save disk space.
Remember to organize the directory structure following the file system tree structure in splits files (which is the original raw files' structure).
Finally, compress everything in a single tarball.

### **NYU**

You can download the dataset from [Google Drive Link](https://drive.google.com/file/d/1wC-io-14RCIL4XTUrQLk6lBqU2AexLVp/view?usp=share_link).
Splits are provided [here](../splits/nyu). For compatibility, we suggest to re-compress with splits files as a tarball.


### **NYU**

You can download the dataset from [link](https://hkuhk-my.sharepoint.com/:f:/g/personal/xjqi_hku_hk/Ek0Vm--5oi1GssioLE5LjO0ByLTKpWAG00zYYUCeiydR7g?e=8kAdLZ) from [GeoNet repo](https://github.com/xjqi/GeoNet).
Splits are provided [here](../splits/nyu_normal).


### **SUNRGBD**

The dataset could be downloaded from [here](https://rgbd.cs.princeton.edu/).
Splits are provided [here](../splits/sunrgbd). For compatibility, we suggest to re-compress with splits files as a tarball.


### **Diode**

The dataset could be downloaded from [here](https://diode-dataset.org/).
Splits are provided [here](../splits/diode). For compatibility, we suggest to re-compress with splits files as a tarball.


### **Argoverse1.1**

Clone [argoverse](https://github.com/argoverse/argoverse-api) repo and export to your `PYTHONPATH`. 
```shell
cd ..
git clone https://github.com/argoverse/argoverse-api
export PYTHONPATH="$PWD/argoverse-api:$PYTHONPATH"
``` 

Then run the code in ``../splits/argo`` to download and process the dataset. You can then use the splits and info files in ``../splits/argo``.
The option ``--split`` refers to downloading and processing the Argoverse splits (namely: train1, train2, train3, train4, val), see original website for more details.
``<BASE-PATH>`` in this example corresponds to the root directory where to download, extract, process and output data
```shell
cd ./idisc
pyhton ./splits/argo/get_argoverse.py --base-path <BASE-PATH> --split <split-chosen>
``` 

### **DDAD**
Clone [DDAD](https://github.com/TRI-ML/DDAD) repo and export to your `PYTHONPATH`. 
```shell
cd ..
git clone https://github.com/TRI-ML/DDAD
export PYTHONPATH="$PWD/dgp:$PYTHONPATH"
``` 

Then run the code in ``../splits/ddad`` to download and process the dataset. You can then use the splits and info files in ``../splits/ddad``
The option ``--split`` refers to downloading and processing the DDAD splits (namely, train or val), see original website for more details.
``<BASE-PATH>``` in this example corresponds to the root directory where to download, extract, process and output data
```shell
cd ./idisc
pyhton ./splits/ddad/get_ddad.py --base-path <BASE-PATH> --split <split-chosen>
``` 
