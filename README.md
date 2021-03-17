## CIA-SSD: Updated version

Currently state-of-the-art single-stage object detector from point cloud on KITTI Benchmark, running with 32FPS.

## Pipeline
![pipeline](images/pipeline.png)
First, input point cloud (a) is encoded with a sparse convolutional network denoted by SPConvNet (b). Then,spatial-semantic feature aggregation (SSFA) module (c) fuses the extracted spatial and semantic features using attentional fusion module (d). After this, the multi-task head (e) realizes the object classification and localization using a confidence function. In the end, the distance-variant IoU-weighted NMS (DI-NMS) is formulated for post-processing.
For more detailed information please refer this [Paper](https://arxiv.org/abs/2012.03015).

## Installation
First, install [Det3D](https://github.com/Otteri/Det3D/blob/master/INSTALLATION.md) dependencies: [spconv](https://github.com/poodarchu/spconv) and [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit). Then, proceed to install CIA-SSD & Det3D:
```bash
$ git clone --recursive git@github.com:Otteri/CIA-SSD.git
$ cd ./CIA-SSD/Det3D/det3d/core/iou3d # TODO: fix this, may not work anymore
$ python setup.py install
$ cd ./CIA-SSD
$ python setup.py build develop
```

### Getting training data
We need data for training. Please, refer Det3D [data preparation guide](https://github.com/Otteri/Det3D/blob/master/GETTING_STARTED.md). Let's consider KITTI dataset. Download the [KITTI data](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and order it as guided. After this, we can use Det3D scripts to prepare the data for us:
```
python Det3D/tools/create_data.py kitti_data_prep --root_path="<KITTI_DATASET_ROOT>"
```
After data preparation, you may want to check that the model configuration is okay:
`examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py`
### Training & Evaluation
Now you should be able to train the model (single GPU):
```
python Det3D/tools/train.py <config>
```
Evaluation scores will be printed when training finishes. However, you can also evaluate the model whenever you like with:
```
python Det3D/tools/test.py <config> <checkpoint>
```

## Acknowledgements
- [CIA-SSD](https://github.com/Vegeta2020/CIA-SSD)
- [Det3D](https://github.com/poodarchu/Det3D)

## License
This codebase is released under the Apache 2.0 license.
