# This file defines KITTI paths, so that required data can be found.
# The data is assumed to be ordered in a following way by default:
#
# KITTI
#   ├── training
#   |   ├── calib
#   |   ├── image_2
#   |   ├── label_2
#   |   ├── planes
#   |   ├── velodyne
#   |   └── velodyne_reduced  # empty directory
#   └── testing
#       ├── calib
#       ├── image_2
#       ├── velodyne
#       └── velodyne_reduced # empty directory

data_root = "/home/cosmo/data/KITTI/"
train_anno = data_root + "/kitti_infos_train.pkl"
val_anno = data_root + "/kitti_infos_val.pkl"
test_anno = data_root + "/kitti_infos_test.pkl"
db_info_path = data_root + "/dbinfos_train.pkl"
