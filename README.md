Implementation of "Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields" for [AI challenger 
keypoint competition](https://challenger.ai/competition/keypoint/subject)

## Train demo

1. install cython package

```
./cython/rebuild.sh

```

2. Generate intermediate files

change folder name and json name in pose_io/parse_label.py

```
path1 = '/data/guest_users/liangdong/liangdong/practice_demo/AIchallenger/keypoint_validation_annotations_20170911.json' 
trainimagepath = '/data/guest_users/liangdong/liangdong/practice_demo/AIchallenger/validation_image/keypoint_validation_images_20170911/'
```

```
python pose_io/parse_label.py 
```

3. Train

```
python TrainWeight.py
```

You can download mxnet model and parameters for vgg19 from [here](http://data.mxnet.io/models/imagenet/vgg/)


### Cite paper Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields

```
@article{cao2016realtime,
  title={Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  author={Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
  journal={arXiv preprint arXiv:1611.08050},
  year={2016}
  }
```

## Other implementations of Realtime Multi-Person 2D Pose Estimation

[Original caffe training model](https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose)

[Original data preparation and demo](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

[Pytorch](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)

[keras](https://github.com/raymon-tian/keras_Realtime_Multi-Person_Pose_Estimation)

[mxnet](https://github.com/dragonfly90/mxnet_Realtime_Multi-Person_Pose_Estimation)
