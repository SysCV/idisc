# Getting Started
This document provides tutorials to train and evaluate iDisc. Before getting started, make sure you have finished [installation](INSTALL.md) and [dataset setup](DATA.md).
For evaluation the data need to be unzipped directly the ``<BASE-PATH>/datasets`` folder. Split files are meant to be dumped in ``<BASE-PATH>/datasets``, too (good practice is to have them directly in the dataset tarball).
``<BASE-PATH>`` is meant to be a temporary file where to store dataset or intermediate outputs (e.g., /tmp or /scatch).
Data structure example with NYU:
```
<BASE-PATH>
└── datasets/
    └── basement_001a/
        ├── rgb_00000.jpg
        .
        .
        └── sync_depth_00200.png
    ├── basement_001b/
        ├── rgb_00000.jpg
        .
        .
        └── sync_depth_00048.png
    .
    .
    ├── nyu_train.txt (split used for training)
    ├── nyu_test.txt (split used for testing)
    .
    .
    └── study_room_0005b/
```

## Evaluation
First, download the models you want to evaluate from our [model zoo](MODEL_ZOO.md).


## Inference

Please first unzip the dataset you want to evaluate on under ``<BASE-PATH>/datasets``. Then, you can launch evaluation with the command:
```shell
python ./scripts/test.py --model-file <MODEL-FILE-PATH> --config-file <CONFIG-FILE-PATH> --base-path <BASE-PATH>
```

`<MODEL-FILE-PATH>` is the path to the model to be tested.<br>
`<CONFIG-FILE-PATH>` is the corresponding config file path.<br>
`<BASE-PATH>` is the root of "datasets" directory (e.g /tmp)

To reproduce our result on, e.g, KITTI, please run the following code:
```shell
tar -xzf <KITTI-DATASET-PATH> -C <BASE-PATH>/datasets
python ./scripts/test.py --model-file <YOUR-MODEL-DIR>/kitti_swinlarge.pt --config-file <IDISC-REPO>/configs/kitti/kitti_swinl.json --base-path <BASE-PATH>
```

To reproduce our result on the zero-shot settings, e.g, NYUv2 to SUN-RGBD, please run the following code:
```shell
tar -xzf <SUNRGBD-DATASET> -C <BASE-PATH>/datasets
python ./scripts/test.py --model-file <MODEL-DIR>/nyu_swinl.pt --config-file ./configs/nyu/nyu_sunrgbd.json --base-path <BASE-PATH>
```

# Training
For training, shell (or sbatch) script takes care of unzipping and managing data.

## Shell
Train on the dataset of your choice with the a config file. This is a shell command, hence it works for interactive jobs.

```shell
bash ${IDISC-REPO}/scripts/shell_run.sh '<DATASET-TARBALLS>' <CONFIG> <BASE-PATH>
```
`'<DATASET-TARBALLS>'` is the list of datasets to be untar, if nothing to untar, you can pass an empty string (i.e., '').<br>
`<CONFIG-FILE-PATH>` is the config file path you want to use to train.<br>
`<BASE-PATH>` is the root path (e.g., /tmp)

If you encounter any problem, e.g., slurm incompatibilities, you can disable the DDP training by not using the ``--distributed`` flag in the python command in ``shell_run.sh`` script.

To launch a slurm job to train iDisc with SWin-Large on KITTI you can run the following:
```shell
sbatch ${IDISC-REPO}/scripts/shell_run.sh '<YOUR-TARBALL-KITTI-PATH>' ${IDISC-REPO}/configs/kitti/kitti_swinl.json ${BASE-PATH}
```


## Submit

Disclaimer: the submission file is based on the slurm system used in our lab; you should adapt it to your use case.
The instructions as the shell training apply here.
To launch a slurm job to train iDisc with SWin-Large on NYUv2 you can run the following:
```shell
sbatch ${IDISC-REPO}/scripts/submit.sh '<YOUR-TARBALL-NYU-PATH>' ${IDISC-REPO}/configs/nyu/nyu_swinl.json ${BASE-PATH}
```


## Disclaimer

The shell scripts work only for tarballs, dataloading only with files dumped directly on disk. If you prefer to use other compressed datasets (e.g., .zip) you have to change the shell scripts. 
If you want to use other kind of storage for your dataset (e.g., .hdf5), you have to change the shell scripts and, more important, the data loading code.