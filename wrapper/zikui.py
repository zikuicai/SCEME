#!/usr/bin/env python

# when exist multiple gpus, need to assign to only one gpu.
# CUDA_VISIBLE_DEVICES=0 python zikui.py

import os
import re
import argparse
import yaml
import json
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from sin_helper import SinHelper     # cudnn 7.0.4
from model_helper import ModelHelper # cudnn 7.6.0
from data_helper import DataHelper

from test_AE_aux import pred_box_trans
from fast_rcnn.config import cfg, cfg_from_file

def to_tensor_gpu(data):
    data = np.expand_dims(data, axis = 0)
    data = np.expand_dims(data, axis = 0)
    return torch.from_numpy(data).cuda().float()


sin_helper = SinHelper()     # cudnn 7.0.4
model_helper = ModelHelper() # cudnn 7.6.0
data = DataHelper()

config = yaml.safe_load(open('config.yml').read())
for target_cls in config['classes']:
    model_helper[target_cls]
    # model_helper_blackbox[target_cls]


# im_cv is raw image: im_cv = cv2.imread(...)
def image_max_loss(im_cv):
    """Get the max loss of all rois given an image

    Args:
        im_cv (np.ndarray): input image, range 0-255
    Returns:
        max_score (float): the max loss
    """
    im = im_cv.astype(np.float32, copy=True)
    im = np.clip(im,0,255)
    im -= cfg.PIXEL_MEANS # zero_mean

    # resize image and record info
    assert len(cfg.TEST.SCALES) == 1
    target_size = cfg.TEST.SCALES[0]
    im_size_min = np.min(im.shape[0:2])
    im_size_max = np.max(im.shape[0:2])
    im_scale = min([float(target_size) / im_size_min,
                    float(cfg.TEST.MAX_SIZE) / im_size_max])
    # the biggest axis should not be more than MAX_SIZE
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    im_info = np.array([im.shape[0],im.shape[1],im_scale], dtype=np.float32)

    # get context profile in relation
    relation, rois, cls_prob, bbox_pred = sin_helper.eval_img(im, im_info)
    pred_boxes = pred_box_trans(rois, cls_prob, bbox_pred, im_info[-1], im_cv.shape)

    orders = [0, 3, 1, 4, 2]
    weights = [1 / 32.0, 1, 1, 1, 1]
    sorted_relation = [relation[i] for i in orders]
    
    cls_pred = np.argmax(cls_prob, axis=1)

    # get the auto-encoder loss on each roi 
    stat = dict()
    stat['join_optm'] = dict()
    for roi_idx in range(len(cls_pred)):
        roi_class = config['classes'][cls_pred[roi_idx]]
        model = getattr(model_helper, roi_class)
        model.eval()
        context_profile = [sorted_relation[idx][roi_idx] * weights[idx] for idx in range(5)]
        context_profile = to_tensor_gpu(context_profile)
        loss = model(context_profile)
        score = float(loss)

        if str(roi_class) not in stat['join_optm']:
                stat['join_optm'][str(roi_class)] = list()
        stat['join_optm'][str(roi_class)].append((score, pred_boxes.tolist()[roi_idx], float(cls_prob[roi_idx, cls_pred[roi_idx]])))

    # get the max loss as the image score
    max_score = 0
    for key in stat['join_optm']:
        for info in stat['join_optm'][key]:
            if info[0] > max_score:
                max_score = info[0]

    return max_score, stat['join_optm']
# open('./stat-miscls-wb-t/{}.json'.format(p_config), 'w').write(json.dumps(stat))
# open(.format(p_config), 'w').write(json.dumps(stat))
# print '[+]{} finished!'.format(p_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get sceme loss for an image')
    parser.add_argument("--root", nargs="?", help="the root folder of the experiment, like '../../phase1/attacks/attack_mmdetection/res_pspm_pos2/compare_seq_oneshot_eps30' ")
    args = parser.parse_args()

    # read wb txt file 
    # eg. im000006_idx2_f8_t4_ap0_wb[1]_bb[1]
    exp_root = args.root
    exp_root = '../../phase1/attacks/attack_mmdetection/res_pspm_pos2/compare_seq_oneshot_eps50'
    sceme_outputs = os.path.join(exp_root, 'sceme_outputs')
    # sceme_outputs = Path(exp_root) / 'sceme_outputs'
    # sceme_outputs.mkdir(parents=True, exist_ok=True)
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

        # read image
        im_path = os.path.join(img_root,im_id+'.jpg')
        im_cv = np.array(Image.open(im_path))

        if ap not in [0,1,2,3,4,5,6,11]:
            # only include sequential attack, oneshot and oneshot pspm
            continue

        if ap == 0:
            # run on the clean image
            im_cv = np.clip(im_cv,0,255)
            max_score,stat = image_max_loss(im_cv)
            
            sceme_txt = open(os.path.join(exp_root, 'sceme_scores'+'.txt'),'a')
            sceme_txt.write('im{}_clean, {}\n'.format(im_id,max_score))
            sceme_txt.close()
            open(os.path.join(sceme_outputs,'im{}_clean.json'.format(im_id)), 'w').write(json.dumps(stat))

        # read perturbation
        pert_path = os.path.join(pert_root,line+'.npy')
        perturbation = np.load(pert_path)
        im_cv = np.clip(im_cv+perturbation,0,255)
        max_score,stat = image_max_loss(im_cv)
        
        sceme_txt = open(os.path.join(exp_root, 'sceme_scores'+'.txt'),'a')
        sceme_txt.write('{}, {}\n'.format(line,max_score))
        sceme_txt.close()
        open(os.path.join(sceme_outputs,'{}.json'.format(line)), 'w').write(json.dumps(stat))


# im_path = '../data/VOCdevkit/VOC2007/JPEGImages/000001.jpg'
# im_cv = np.array(Image.open(im_path))
# # perturbation = np.array(Image.open(pert_path))
# perturbation = np.zeros_like(im_cv)
# im_cv = np.clip(im_cv+perturbation,0,255)

# max_score,stat = image_max_loss(im_cv)
# print max_score
# print stat

