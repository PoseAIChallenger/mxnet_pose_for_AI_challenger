### Reimplementation of human keypoint detection in mxnet

1. You can download mxnet model and parameters(coco and MPII) from google drive:

   https://drive.google.com/drive/folders/0BzffphMuhDDMV0RZVGhtQWlmS1U

   or check caffe_to_mxnet folder to download original caffe model and transfer it to mxnet model.

2. Test demo based on model of coco dataset: testModel.ipynb
3. Test demo based on model of MPII dataset: testModel_mpi.ipynb
4. Train demo: TrainWeight.py
5. Evaluation on coco validation dataset : evaluation_coco.py

### Cite paper Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields

```
@article{cao2016realtime,
  title={Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  author={Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
  journal={arXiv preprint arXiv:1611.08050},
  year={2016}
  }
```

original caffe training https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose



## TODO:
- [x] Test demo
- [x] Train demo
- [x] Add image augmentation: rotation, flip
- [x] Add weight vector
- [x] Train all images
- [x] Train from vgg model
- [x] evaluation code
- [ ] image read and augmentation in C++

## Other implementations 

[Original caffe training model](https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose)

[Original data preparation and demo](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

[Pytorch](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)

[keras](https://github.com/raymon-tian/keras_Realtime_Multi-Person_Pose_Estimation)
# vgg_cpm
