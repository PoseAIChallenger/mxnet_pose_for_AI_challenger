#from pycocotools.coco import COCO
import sys
sys.path.append('/data/guest_users/liangdong/liangdong/practice_demo')
from collections import namedtuple
import numpy as np
from numpy import ma
import math
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib
import pylab
import json
import time
import mxnet as mx
from numpy import linalg as LA
import cv2
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

path1 = '/data/guest_users/liangdong/liangdong/practice_demo/AIchallenger/keypoint_validation_annotations_20170911.json' #'/home/zhangchenghao/mxnet_cpm/pose_io/AI_label.json'
trainimagepath = '/data/guest_users/liangdong/liangdong/practice_demo/AIchallenger/validation_image/keypoint_validation_images_20170911/'

jointall = []
count = 1
joint_all = dict()


f = open(path1)
label = json.load(f)
i = 0
numimgs = 0
for img in label:
    numimgs += 1


    img_id = img['image_id']
    box  = img['human_annotations']
    keypoint = img['keypoint_annotations']

    if i % 1000 == 0:
        print('processing image:'+img_id)

    temp = cv2.imread(trainimagepath+img_id+'.jpg')
    sp = temp.shape
    height = sp[0]
    width = sp[1]

    prev_center = list(list())


    j = 0
    for human in box:
    	j += 1 # people index
    	person_center = [box[str(human)][0] + box[str(human)][2]/2, box[str(human)][1] + box[str(human)][3]/2 ]

    	flag = 0
        isValidation = 0

        for k in range(len(prev_center)):
            dist1 = prev_center[k][0] - person_center[0]
            dist2 = prev_center[k][1] - person_center[1]
            #print dist1, dist2
            if dist1*dist1+dist2*dist2 < prev_center[k][2]*0.3:
                flag = 1
                continue

        currentjoin={'isValidation': isValidation,
                     'img_paths': trainimagepath + img_id + '.jpg',
                     'objpos': person_center,
                     'image_id': numimgs,
                     'bbox': box[str(human)],
                     'img_width': width,
                     'img_height': height,
                     'segment_area': {},
                     'num_keypoints': 14,
                     'joint_self': np.zeros((14,3)).tolist(),
                     'scale_provided': box[str(human)][3]/368.0,
                     'segmentations': {},
                     'joint_others': {},
                     'annolist_index': numimgs ,
                     'people_index': j,
                     'numOtherPeople':0,
                     'scale_provided_other':{},
                     'objpos_other':{},
                     'bbox_other':{},
                     'segment_area_other':{},
                     'num_keypoints_other':{}
                    }    


        for part in range(14):
            currentjoin['joint_self'][part][0] = keypoint[str(human)][part*3]
            currentjoin['joint_self'][part][1] = keypoint[str(human)][part*3+1]
            # COCO dataset: 1 visible, 2 non-visible
            # 2 means cropped, 0 means occluded by still on image
            if(keypoint[str(human)][part*3+2] == 2):
                currentjoin['joint_self'][part][2] = 1
            elif(keypoint[str(human)][part*3+2] == 1):
                currentjoin['joint_self'][part][2] = 0                
            else:
                currentjoin['joint_self'][part][2] = 2



        count_other = 1
        
        currentjoin['joint_others'] ={}

        for other_human in box:

            if other_human == human:
        	continue

            currentjoin['scale_provided_other'][count_other] = box[str(other_human)][3]/368
            currentjoin['objpos_other'][count_other] = [box[str(other_human)][0]+box[str(other_human)][2]/2, 
                                        box[str(other_human)][1]+box[str(other_human)][3]/2]


            currentjoin['bbox_other'][count_other] = box[str(other_human)]

            currentjoin['num_keypoints_other'][count_other] = 14

            currentjoin['joint_others'][count_other] = np.zeros((14,3)).tolist()

            for part in range(14):
            	currentjoin['joint_others'][count_other][part][0] = keypoint[str(other_human)][part*3]
                currentjoin['joint_others'][count_other][part][1] = keypoint[str(other_human)][part*3+1]
                
                if(keypoint[str(other_human)][part*3+2] == 2):
                    currentjoin['joint_others'][count_other][part][2] = 1
                elif(keypoint[str(other_human)][part*3+2] == 1):
                    currentjoin['joint_others'][count_other][part][2] = 0
                else:
                    currentjoin['joint_others'][count_other][part][2] = 2



            currentjoin['numOtherPeople'] = len(currentjoin['joint_others'])
                    
            count_other = count_other + 1


        prev_center.append([person_center[0], person_center[1],
                            max(box[str(human)][2], box[str(human)][3])])


    joint_all[i] = currentjoin

    count = count + 1

    i += 1


print len(joint_all)


with open('AI_data_val.json', 'w') as f1:
     json.dump(joint_all, f1)


print('-----finish------')







