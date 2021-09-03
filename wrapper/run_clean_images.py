#!/usr/bin/env python

# when exist multiple gpus, need to assign to only one gpu.
# CUDA_VISIBLE_DEVICES=0 python zikui.py

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import re
import argparse
import yaml
import json


import tensorflow
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from zikui import image_max_loss


exp_root = '../../phase1/attacks/attack_mmdetection/res_pspm_pos2/compare_seq_oneshot_eps30'
sceme_outputs = os.path.join(exp_root, 'sceme_outputs_clean')
if not os.path.exists(sceme_outputs):
    os.makedirs(sceme_outputs)

# get all test ids
txt_path = "../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt"
with open(txt_path,'r') as f:
    data = f.readlines()
ids = [i[:-1] for i in data]

img_root = '../data/VOCdevkit/VOC2007/JPEGImages/'
for im_id in tqdm(ids):
    im_path = os.path.join(img_root,im_id+'.jpg')
    im_cv = np.array(Image.open(im_path))

    im_cv = np.clip(im_cv,0,255)
    max_score,stat = image_max_loss(im_cv)

    sceme_txt = open(os.path.join(exp_root, 'sceme_scores_clean'+'.txt'),'a')
    sceme_txt.write('im{}_clean, {}\n'.format(im_id,max_score))
    sceme_txt.close()
    open(os.path.join(sceme_outputs,'im{}_clean.json'.format(im_id)), 'w').write(json.dumps(stat))

