# iDisc Model Zoo and Baselines

## Introduction
Here is the collection of all models whose namespace is compatible with the current repo. We store the output predictions with the same relative path as the depth path from the corresponding dataset. For evaluation we used micro averaging, other repo might use macro averaging, the difference is in the order of decimals of percentage points, but we found it more appropriate for dataset with uneven density distributions due to, e.g., pointcloud accumulation.
Please note that the depth is rescaled as in the original dataset to be stored as .png. In particular to obtain metric depth you need to divide NYUv2 results by 1000, and all other datasets by 256. Normals need to be rescaled from ``[0, 255]`` to ``[-1, 1]``. 
Predictions are not interpolated, that is the output shape is one quarter of input shape.

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

