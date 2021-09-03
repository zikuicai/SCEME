#!/usr/bin/env python

# when exist multiple gpus, need to assign to only one gpu.
# CUDA_VISIBLE_DEVICES=0 python zikui.py

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
sceme_outputs = os.path.join(exp_root, 'sceme_outputs_noclip')
if not os.path.exists(sceme_outputs):
    os.makedirs(sceme_outputs)

pert_root = os.path.join(exp_root, 'pert')
img_root = '../data/VOCdevkit/VOC2007/JPEGImages/'

exp_name = exp_root.split('/')[-1]
wb_file = os.path.join(exp_root, '{}.txt'.format(exp_name))
wb_txt = open(wb_file, 'r').readlines()
for line in tqdm(wb_txt):
    line = line.strip()
    im_id = re.findall(r"im(.+?)\_", line)[0]
    idx = re.findall(r"idx(.+?)\_", line)[0]
    from_class = re.findall(r"f(.+?)\_", line)[0]
    to_class = re.findall(r"t(.+?)\_", line)[0]
    ap = int(re.findall(r"ap(.+?)\_", line)[0])

    # only run on attack images
    if ap != 11:
        continue

    # read image
    im_path = os.path.join(img_root,im_id+'.jpg')
    im_cv = np.array(Image.open(im_path))

    # read perturbation
    pert_path = os.path.join(pert_root,line+'.npy')
    perturbation = np.load(pert_path)
    # im_cv = np.clip(im_cv+perturbation,0,255)
    im_cv = im_cv+perturbation
    max_score,stat = image_max_loss(im_cv)
    
    sceme_txt = open(os.path.join(exp_root, 'sceme_scores_noclip'+'.txt'),'a')
    sceme_txt.write('{}, {}\n'.format(line,max_score))
    sceme_txt.close()
    open(os.path.join(sceme_outputs,'{}.json'.format(line)), 'w').write(json.dumps(stat))

