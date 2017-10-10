import sys
import os
import mxnet as mx
import numpy as np
import copy
import re
import json
from google.protobuf import text_format
import json
import cv2 as cv
import scipy
import PIL.Image
import math
import time
import scipy
import matplotlib
import pylab as plt
from scipy.ndimage.filters import gaussian_filter
from modelCPMWeight import *
from collections import namedtuple
Point = namedtuple('Point', 'x y')
crop_size_x = 368
crop_size_y = 368
center_perterb_max = 40

scale_prob = 1
scale_min = 0.5
scale_max = 1.1
target_dist = 0.6


output_prefix='model/testConfigModel'
sym, arg_params, aux_params = mx.model.load_checkpoint(output_prefix, 44)

csym = CPMModel_test()

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

class DataBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = [data]
        self.label = [label]
        self.pad = pad

def applyDNN(oriImg, images, sym1, arg_params1, aux_params1):
    
    imageToTest_padded, pad = padRightDownCorner(images, 8, 128)
    transposeImage = np.transpose(np.float32(imageToTest_padded[:,:,:]), (2,0,1))/256 - 0.5
    testimage = transposeImage
    # print testimage.shape
    cmodel = mx.mod.Module(symbol=csym, label_names=[], context=mx.gpu(1))
    # print testimage.shape
    cmodel.bind(data_shapes=[('data', (1, 3, testimage.shape[1], testimage.shape[2]))])
    cmodel.init_params(arg_params=arg_params1, aux_params=aux_params1)
    # print 'init_params failed'
    onedata = DataBatch(mx.nd.array([testimage[:,:,:]]), 0)
    #print 'batch'
    cmodel.forward(onedata)
    #print 'forward'
    result = cmodel.get_outputs()
    
    heatmap = np.moveaxis(result[1].asnumpy()[0], 0, -1)
    heatmap = cv.resize(heatmap, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
    
    pagmap = np.moveaxis(result[0].asnumpy()[0], 0, -1)
    pagmap = cv.resize(pagmap, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
    pagmap = pagmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    pagmap = cv.resize(pagmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
    
    # print heatmap.shape
    # print pagmap.shape
    return heatmap, pagmap

def applyModel(oriImg, param, sym, arg_params, aux_params):
    model = param['model']
    model = model[param['modelID']]
    boxsize = model['boxsize']
    makeFigure = 0 
    numberPoints = 1
    octave = param['octave']
    starting_range = param['starting_range']
    ending_range = param['ending_range'] 
    boxsize = 368
    scale_search = [0.5, 1, 1.5, 2]
    multiplier = [x * boxsize*1.0/ oriImg.shape[0] for x in scale_search] 
  
    
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], numofparts))
    pag_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], numoflinks*2))
    for i in range(len(multiplier)):
        # print i
        cscale = multiplier[i]
        imageToTest = cv.resize(oriImg, (0,0), fx=cscale, fy=cscale, interpolation=cv.INTER_CUBIC)
        
        heatmap, pagmap = applyDNN(oriImg, imageToTest, sym, arg_params, aux_params)
        # print(heatmap.shape)
        # print(pagmap.shape)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        pag_avg = pag_avg + pagmap / len(multiplier)
        # print 'add one layer'
    return heatmap_avg, pag_avg





modelId = 1
# set this part
param = dict()
# GPU device number (doesn't matter for CPU mode)
GPUdeviceNumber = 0
# Select model (default: 5)
param['modelID'] = modelId
# Use click mode or not. If yes (1), you will be asked to click on the center
# of person to be pose-estimated (for multiple people image). If not (0),
# the model will simply be applies on the whole image.
param['click'] = 1
# Scaling paramter: starting and ending ratio of person height to image
# height, and number of scales per octave
# warning: setting too small starting value on non-click mode will take
# large memory
# CPU mode or GPU mode
param['use_gpu'] = 1
param['test_mode'] = 3
param['vis'] = 1
param['octave'] = 6
param['starting_range'] = 0.8
param['ending_range'] = 2
param['min_num'] = 4
param['mid_num'] = 10
# the larger the crop_ratio, the smaller the windowsize
param['crop_ratio'] = 2.5  # 2
param['bbox_ratio'] = 0.25 # 0.5
# applyModel_max
param['max'] = 0
# use average heatmap
param['merge'] = 'avg'

# path of your caffe
caffepath = '/home/zhecao/caffe/matlab';

if modelId == 1:
    param['scale_search'] = [0.5, 1, 1.5, 2]
    param['thre1'] = 0.1
    param['thre2'] = 0.05 
    param['thre3'] = 0.5 

    param['model'] = dict()
    param['model'][1] = dict()
    param['model'][1]['caffemodel'] = '../model/_trained_COCO/pose_iter_440000.caffemodel'
    param['model'][1]['deployFile'] = '../model/_trained_COCO/pose_deploy.prototxt'
    param['model'][1]['description'] = 'COCO Pose56 Two-level Linevec'
    param['model'][1]['boxsize'] = 368
    param['model'][1]['padValue'] = 128
    param['model'][1]['np'] = 18
    param['model'][1]['part_str'] = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 
                             'Lsho', 'Lelb', 'Lwri', 
                             'Rhip', 'Rkne', 'Rank',
                             'Lhip', 'Lkne', 'Lank',
                             'Leye', 'Reye', 'Lear', 'Rear', 'pt19']
if modelId == 2:
    param['scale_search'] = [0.7, 1, 1.3]
    param['thre1'] = 0.05
    param['thre2'] = 0.01 
    param['thre3'] = 3
    param['thre4'] = 0.1

    param['model'] = dict()
    param['model'][2] = dict()
    param['model'][2]['caffemodel'] = '../model/_trained_MPI/pose_iter_146000.caffemodel'
    param['model'][2]['deployFile'] = '../model/_trained_MPI/pose_deploy.prototxt'
    param['model'][2]['description'] = 'COCO Pose56 Two-level Linevec'
    param['model'][2]['boxsize'] = 368
    param['model'][2]['padValue'] = 128
    param['model'][2]['np'] = 18
    param['model'][2]['part_str'] = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri',  
                                     'Lsho', 'Lelb', 'Lwri', 
                                     'Rhip', 'Rkne', 'Rank', 
                                     'Lhip', 'Lkne', 'Lank', 
                                     'Leye', 'Reye', 'Lear', 'Rear', 'pt19']


mid_1 = [1, 2, 4, 5, 1, 7, 8,  4, 10, 11, 13, 1, 4]
mid_2 = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 14, 14]
newslist = [[x,y] for x, y in zip(mid_1, mid_2)]


mapidx1 = range(1, 27, 2)
mapidx2 = range(2, 28, 2)
mapid = [[x,y] for x, y in zip(mapidx1, mapidx2)]
mapid = mapid[3:]+mapid[0:3]


# test AI challenge validation dataset
# rootdir= '../ai_dataset/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911/'
rootdir = '../ai_dataset/ai_challenger_keypoint_test_a_20170923/keypoint_test_a_images_20170923/'
images = os.listdir(rootdir)
count = 0

joints_all = []

for image in images:
    count += 1
    cimage = cv.imread(rootdir + str(image))


    heatmap_avg, paf_avg = applyModel(cimage, param, sym, arg_params, aux_params)


    all_peaks = []
    peak_counter = 0

    for part in range(14):
        x_list = []
        y_list = []
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        cid = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (cid[i],) for i in range(len(cid))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)


    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = newslist
    # the middle joints heatmap correpondence
    mapIdx = mapid

    connection_all = []
    special_k = []
    special_non_zero_index = []
    mid_num = 11

    for k in range(len(mapIdx)):
        
        score_mid = paf_avg[:,:,[x-1 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
        # print(k)
        # print(candA)
        # print('---------')
        # print(candB)
        # print limbSeq[k][0], limbSeq[k][1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    # print('vec: ',vec)
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
		    if norm == 0:
			norm = 0.1
                    # print('norm: ', norm)
                    vec = np.divide(vec, norm)
                    # print('normalized vec: ', vec)
                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num))
                    # print('startend: ', startend)
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    # print('vec_x: ', vec_x)
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])
                    # print('vec_y: ', vec_y)
                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    # print(score_midpts)
                    # print('score_midpts: ', score_midpts)
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*cimage.shape[0]/norm-1, 0)

                    # print('score_with_dist_prior: ', score_with_dist_prior)
                    criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.5 * len(score_midpts)
                    # print('score_midpts > param["thre2"]: ', len(np.nonzero(score_midpts > param['thre2'])[0]))
                    criterion2 = score_with_dist_prior > 0

                    '''
                    if k==1 or k==2:
                        criterion2 = score_with_dist_prior > 1
                    if k == 2:
                        print '-----------', i, j
                        print criterion1
                        print criterion2
                        print score_with_dist_prior
                        print score_midpts
                    '''
                    if criterion1 and criterion2:
                        # print('match')
                        # print(i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2])
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])
                    # print('--------end-----------')
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            # print('-------------connection_candidate---------------')
            # print(connection_candidate)
            # print('------------------------------------------------')
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    # print('----------connection-----------')
                    # print(connection)
                    # print('-------------------------------')
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        elif(nA != 0 or nB != 0):
            #print 'k: ', special_k
            special_k.append(k)
            special_non_zero_index.append(indexA if nA != 0 else indexB)
            connection_all.append([])

    #print connection_all
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 16))

    candidate = np.array([item for sublist in all_peaks for item in sublist])

    # print len(connection_all)
    # print len(mapIdx)
    print 'special_k:', special_k
    for k in range(len(mapIdx)):
        if k not in special_k:
            
            try:
                partAs = connection_all[k][:,0]
                partBs = connection_all[k][:,1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])): #= 1:size(temp,1)
                    
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)): #1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        # if k==2:
                            # print 'j ', j
                        if subset[j][indexB] != partBs[i]:
                            # if k==2:
                                # print 'j ', j
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2: # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        # print "found = 2"
                        membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0: #merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else: # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 13:
                        row = -1 * np.ones(16)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
                    # if k==2:
                        # print 'k ', found
            except:
                print k
            
    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    
    for i in range(len(subset)):
        if subset[i][-1] < 5 or subset[i][-2]/subset[i][-1] < 0.2:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)


    keypoint = dict()

    for i in range(subset.shape[0]):
        joints = [0 for k in range(14*3)]
        for j in range(len(subset[i])-2):
            if int(subset[i][j]) == -1:
                joints[j*3] = 0
                joints[j*3+1] = 0
                joints[j*3+2] = 0
            else:
                for m in range(14):
                    for n in range(len(all_peaks[m])):
                        if int(subset[i][j]) == all_peaks[m][n][3]:
                            #print([all_peaks[m][n][0], all_peaks[m][n][1], 1])
                            joints[j*3] = int(all_peaks[m][n][0])
                            joints[j*3+1] = int(all_peaks[m][n][1])
                            joints[j*3+2] = 1
                            break
        keypoint['human'+str(subset.shape[0]-i)] = joints


    # sort keypoint
    current_joints = {'image_id':image[:-4], 'keypoint_annotations':keypoint}
    joints_all.append(current_joints)

    
    print 'steps:', count


with open('AI_pose_mxnet.json', 'w') as f:
    json.dump(joints_all, f)


