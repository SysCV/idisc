[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/idisc-internal-discretization-for-monocular/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=idisc-internal-discretization-for-monocular)
[![KITTI Benchmark](https://img.shields.io/badge/KITTI%20Benchmark-3rd%20among%20all%20at%20submission%20time-blue)](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/idisc-internal-discretization-for-monocular/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=idisc-internal-discretization-for-monocular)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/idisc-internal-discretization-for-monocular/surface-normals-estimation-on-nyu-depth-v2-1)](https://paperswithcode.com/sota/surface-normals-estimation-on-nyu-depth-v2-1?p=idisc-internal-discretization-for-monocular)


# iDisc: Internal Discretization for Monocular Depth Estimation

![](docs/idisc-banner.png)

> [**iDisc: Internal Discretization for Monocular Depth Estimation**](),            
> Luigi Piccinelli, Christos Sakaridis, Fisher Yu,
> CVPR 2023 (to appear)
> *Project Website ([iDisc](http://vis.xyz/pub/idisc/))* 
> *Paper ([arXiv 2304.06334](https://arxiv.org/pdf/2304.06334.pdf))*


## Visualization

### KITTI
<p align="center">
  <img src="docs/kitti_example.gif" alt="animated" />
</p>


### NYUv2-Depth
<p align="center">
  <img src="docs/nyu_example.gif" alt="animated" />
</p>

For more, and not compressed, visual examples please visit [vis.xyz](http://vis.xyz/pub/idisc/).

## Citation

If you find our work useful in your research please consider citing our publication:
```bibtex
    @inproceedings{piccinelli2023idisc,
      title={iDisc: Internal Discretization for Monocular Depth Estimation},
      author={Piccinelli, Luigi and Sakaridis, Christos and Yu, Fisher},
      booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2023}
    }
```


## Abstract
Monocular depth estimation is fundamental for 3D scene understanding and downstream applications. However, even under the supervised setup, it is still challenging and ill posed due to the lack of geometric constraints. We observe that although a scene can consist of millions of pixels, there are much fewer high-level patterns. We propose iDisc to learn those patterns with internal discretized representations. The method implicitly partitions the scene into a set of high-level concepts. In particular, our new module, Internal Discretization (ID), implements a continuous-discrete-continuous bottleneck to learn those concepts without supervision. In contrast to state-of-the-art methods, the proposed model does not enforce any explicit constraints or priors on the depth output. The whole network with the ID module can be trained in an end-to-end fashion thanks to the bottleneck module based on attention. Our method sets the new state of the art with significant improvements on NYU-Depth v2 and KITTI, outperforming all published methods on the official KITTI benchmark. iDisc can also achieve state-of-the-art results on surface normal estimation. Further, we explore the model generalization capability via zero-shot testing. From there, we observe the compelling need to promote diversification in the outdoor scenario and we introduce splits of two autonomous driving datasets, DDAD and Argoverse


## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and to [DATA.md](docs/DATA.md) for datasets preparation.


## Get Started

Please see [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the basic usage of iDisc.


## Model Zoo


### General

We store the output predictions in the same relative path as the depth path from the corresponding dataset. For evaluation we used micro averaging, while some other depth repos use macro averaging; the difference is in the order of decimals of percentage points, but we found it more appropriate for datasets with uneven density distributions, e.g. due to point cloud accumulation or depth cameras.
Please note that the depth map is rescaled as in the original dataset to be stored as .png file. In particular, to obtain metric depth, you need to divide NYUv2 results by 1000, and results for all other datasets by 256. Normals need to be rescaled from ``[0, 255]`` to ``[-1, 1]``. 
Predictions are not interpolated, that is, the output dimensions are one quarter of the input dimensions. For evaluation we used bilinear interpolation with aligned corners.


## KITTI

| Backbone | d0.5 | d1 | d2 | RMSE | RMSE log | A.Rel | Sq.Rel | Config | Weights | Predictions |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Resnet101 | 0.860 | 0.965 | 0.996 | 2.362 | 0.090 | 0.059 | 0.197 | [config](configs/kitti/kitti_r101.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/kitti_resnet101.pt) | [predictions](https://dl.cv.ethz.ch/idisc/predictions/kitti_resnet101.tar) |
| EfficientB5 |0.852 | 0.963 | 0.994 | 2.510 | 0.094 | 0.063 | 0.223 | [config](configs/kitti/kitti_eb5.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/kitti_effnetb5.pt) | [predictions](https://dl.cv.ethz.ch/idisc/predictions/kitti_effnetb5.tar) |
| Swin-Tiny | 0.870 | 0.968 | 0.996 | 2.291 | 0.087 | 0.058 | 0.184 | [config](configs/kitti/kitti_swint.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/kitti_swintiny.pt) | [predictions](https://dl.cv.ethz.ch/idisc/predictions/kitti_swintiny.tar) |
| Swin-Base | 0.885 | 0.974 | 0.997 | 2.149 | 0.081 | 0.054 | 0.159 | [config](configs/kitti/kitti_swinb.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/kitti_swinbase.pt) | [predictions](https://dl.cv.ethz.ch/idisc/predictions/kitti_swinbase.tar) |
| Swin-Large | 0.896 | 0.977 | 0.997 | 2.067 | 0.077 | 0.050 | 0.145 | [config](configs/kitti/kitti_swinl.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/kitti_swinlarge.pt) | [predictions](https://dl.cv.ethz.ch/idisc/predictions/kitti_swinlarge.tar) |


## NYUv2

| Backbone | d1 | d2 | d3 | RMSE | A.Rel | Log10 | Config | Weights | Predictions |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Resnet101 | 0.892 | 0.983 | 0.995 | 0.380 | 0.109 | 0.046 | [config](configs/nyu/nyu_r101.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/nyu_resnet101.pt) | [predictions](https://dl.cv.ethz.ch/idisc/predictions/nyu_resnet101.tar) |
| EfficientB5 | 0.903 | 0.986 | 0.997 | 0.369 | 0.104 | 0.044 | [config](configs/nyu/nyu_eb5.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/nyu_effnetb5.pt) | [predictions](https://dl.cv.ethz.ch/idisc/predictions/nyu_effnetb5.tar) |
| Swin-Tiny | 0.894 | 0.983 | 0.996 | 0.377 | 0.109 | 0.045 | [config](configs/nyu/nyu_swint.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/nyu_swintiny.pt) | [predictions](https://dl.cv.ethz.ch/idisc/predictions/nyu_swintiny.tar) |
| Swin-Base | 0.926 | 0.989 | 0.997 | 0.327 | 0.091 | 0.039 | [config](configs/nyu/nyu_swinb.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/nyu_swinbase.pt) | [predictions](https://dl.cv.ethz.ch/idisc/predictions/nyu_swinbase.tar) |
| Swin-Large | 0.940 | 0.993 | 0.999 | 0.313 | 0.086 | 0.037 | [config](configs/nyu/nyu_swinl.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/nyu_swinlarge.pt) | [predictions](https://dl.cv.ethz.ch/idisc/predictions/nyu_swinlarge.tar) |


### Normals

Results may differ (~0.1%) due to micro vs. macro averaging and bilinear vs. bicubic interpolation.

| Backbone | 11.5 | 22.5 | 30 | RMSE | Mean | Median | Config | Weights | Predictions |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Swin-Large | 0.637 | 0.796 | 0.855 | 22.9 | 14.6 | 7.3 | [config](configs/nyunorm/nyunorm_swinl.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/nyunormals_swinlarge.pt) | [predictions](https://dl.cv.ethz.ch/idisc/predictions/nyunormals_swinlarge.tar) |
 

## DDAD 

| Backbone | d1 | d2 | d3 | RMSE | RMSE log | A.Rel | Sq.Rel | Config | Weights | Predictions |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Swin-Large | 0.809 | 0.934 | 0.971 | 8.989 | 0.221 | 0.163 | 1.85 | [config](configs/ddad/ddad_swinl.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/ddad_swinlarge.pt) | [predictions](https://dl.cv.ethz.ch/idisc/predictions/ddad_swinlarge.tar) |


## Argoverse

| Backbone | d1 | d2 | d3 | RMSE | RMSE log | A.Rel | Sq.Rel | Config | Weights | Predictions | 
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Swin-Large | 0.821 | 0.923 | 0.960 | 7.567 | 0.243 | 0.163 | 2.22 | [config](configs/argo/argo_swinl.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/argo_swinlarge.pt) | [predictions](https://dl.cv.ethz.ch/idisc/predictions/argo_swinlarge.tar) |


### Zero-shot testing

|Train Dataset| Test Dataset | d1 | RMSE | A.Rel | Config | Weights |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| NYUv2 | SUN-RGBD | 0.838 |  0.387 | 0.128 | [config](configs/nyu/nyu_sunrgbd.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/nyu_swinlarge.pt) |
| NYUv2 | Diode | 0.810 |  0.721 | 0.156 | [config](configs/nyu/nyu_diode.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/nyu_swinlarge.pt) |
| KITTI | Argoverse | 0.560 |  12.18 | 0.269 |  [config](configs/kitti/kitti_argo.json) | [weights](https://dl.cv.ethz.ch/idisc/checkpoints/kitti_swinlarge.pt) |
| KITTI | DDAD | 0.350 |  14.26 | 0.367 |  [config](configs/kitti/kitti_ddad.json) |  [weights](https://dl.cv.ethz.ch/idisc/checkpoints/kitti_swinlarge.pt) |


## License

This software is released under Creatives Common BY-NC 4.0 license. You can view a license summary [here](LICENSE).


## Contributions

If you find any bug in the code, please report to <br>
Luigi Piccinelli (lpiccinelli_at_ethz.ch)


## Acknowledgement

This work is funded by Toyota Motor Europe via the research project [TRACE-Zurich](https://trace.ethz.ch) (Toyota Research on Automated Cars Europe).
