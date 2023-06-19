
## Installation

### Requirements
- Linux
- Python 3.9+ 
- PyTorch 1.10+
- CUDA 11.0+
- NCCL 2+
- GCC 5+
Requirements are not in principle hard requirements, since we are using basic pytorch/torchvision operations, but there might be some differences (not tested).

### Install iDisc

a. Create a conda virtual environment and activate it.
```shell
conda create -n idisc python=3.9 -y
conda activate idisc
```

b. Install PyTorch and torchvision,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).


c. Install IDisc addiitonal requirements and export to ``PYTHONPATH`` (add to .bashrc in order not to export in every new shell)
```shell
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

d. Install Deformable Attention
```shell
export IDISC-REPO=${PWD}
cd ${IDISC-REPO}/idisc/models/ops/
bash ./make.sh
```
