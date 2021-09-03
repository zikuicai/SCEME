# Import useless modele to force the process use cudnn 
# comes from tensorflow instead of torch
import tensorflow
import torch
# Do not delete this

from sin_helper import SinHelper     # cudnn 7.0.4
from model_helper import ModelHelper # cudnn 7.6.0
from data_helper import DataHelper

from utils.timer import Timer

import pickle
import json
import os

import numpy as np

timer = Timer()

def create_mask(shape, box, content=None):
	mask = np.zeros([int(shape[0]), int(shape[1]), 3])
	x1, y1, x2, y2 = np.array(box[:4], dtype=np.int)
	if content is not None:
		mask[y1:y2,x1:x2] = content
	else:
		mask[y1:y2,x1:x2] = 1
	return mask 

def to_tensor_gpu(data):
    data = np.expand_dims(data, axis = 0)
    data = np.expand_dims(data, axis = 0)
    return torch.from_numpy(data).cuda().float()

def get_perturbation(folder='./IFGSM_p_womask_miscls', filename_filter=''):
    assert os.path.exists(folder), 'Folder does not exist'
    for i in os.listdir(folder):
        if filename_filter != '' and filename_filter is not None:
            if filename_filter not in i:
                continue
        yield (
         i, pickle.load(open(os.path.join(folder, i))))

    return

def main():
    sin_helper = SinHelper()     # cudnn 7.0.4
    model_helper = ModelHelper() # cudnn 7.6.0
    model_helper_blackbox = ModelHelper('1.0')
    data = DataHelper()

    import yaml
    config = yaml.safe_load(open('config.yml').read())
    for target_cls in config['classes']:
        model_helper[target_cls]
        model_helper_blackbox[target_cls]

    from fast_rcnn.config import cfg, cfg_from_file

    print('all loaded success!')

    import code

    from test_AE_aux import pred_box_trans

    # perturbation = pickle.load(open('/data/ml-hdd/IFGSM_p_womask_miscls_tmp2/im313_box0_f7_t12'))

    # im += perturbation
    for p_config, perturbation in get_perturbation(config['SIN']['perturbation_path']):
<<<<<<< HEAD
=======
        # if p_config != 'im383_box0_f3_t11':
        #     continue
        # if os.path.exists('./stat-miscls-wb/{}.json'.format(p_config)):
        #     print('{} exists, skipping...'.format(p_config))
        #     continue
>>>>>>> 8251082763a83acfdb087aca572b97621c4543f0
        perturbation = np.clip(perturbation, -10, 10)
        stat = dict()
        
        t = p_config.split('_')
        file_name_id = int(t[0][2:])
        box_id = eval(t[1][3:].replace('-',','))
        # box_id = int(t[1][3:])
        # f_id = int(t[2][1:])
        t_id = int(t[3][1:])

<<<<<<< HEAD

=======
>>>>>>> 8251082763a83acfdb087aca572b97621c4543f0
        im_cv, orig_im, im_info, gt_boxes = data.get_image_by_name(file_name_id)

        im = orig_im.copy()

        relation, rois, cls_prob, bbox_pred = 1,2,3,4 
        # relation, rois, cls_prob, bbox_pred = sin_helper.eval_img(im, im_info)

        orders = [0, 3, 1, 4, 2]
        weights = [1 / 32.0, 1, 1, 1, 1]
        # sorted_relation = [relation[i] for i in orders]


        # pred_boxes = pred_box_trans(rois, cls_prob, bbox_pred, im_info[-1], im_cv.shape)

        # orig_cls_pred = np.argmax(cls_prob, axis=1)

        # stat['original'] = dict()

        # for roi_idx in range(len(orig_cls_pred)):
        #     roi_class = config['classes'][orig_cls_pred[roi_idx]]
        #     # print(roi_class)
        #     model = getattr(model_helper, roi_class)
        #     model.eval()
        #     context_profile = [sorted_relation[idx][roi_idx] * weights[idx] for idx in range(5)]
        #     context_profile = to_tensor_gpu(context_profile)
        #     # print(roi_class, float(model(context_profile))
        #     if str(roi_class) not in stat['original']:
        #         stat['original'][str(roi_class)] = list()
        #     stat['original'][str(roi_class)].append(float(model(context_profile)))

# add perturbations
        im += perturbation
        # pickle.dump(im, open('vanilla.npy','w'))
# """""""""""""""""""""""""""""""""
#     eval in test net for comparison
# """""""""""""""""""""""""""""""""
        # print 'start eval in test net'
        relation, rois, cls_prob, bbox_pred = sin_helper.eval_img(im, im_info)
        # print 'finish eval in test net'
        
        sorted_relation = [relation[i] for i in orders]

        pred_boxes = pred_box_trans(rois, cls_prob, bbox_pred, im_info[-1], im_cv.shape)
        orig_cls_pred = np.argmax(cls_prob, axis=1)

        stat['perturbed'] = dict()

        # target_context_profile = [np.empty( (256, 4096) ,dtype=float) for _ in range(5)]
        for roi_idx in range(len(orig_cls_pred)):
            roi_class = config['classes'][orig_cls_pred[roi_idx]]
            # print(roi_class)
            model = getattr(model_helper, roi_class)
            model.eval()
            context_profile = [sorted_relation[idx][roi_idx] * weights[idx] for idx in range(5)]
            context_profile = to_tensor_gpu(context_profile)
            # print(roi_class, float(model(context_profile))
            loss = model(context_profile)
            score = float(loss)
            
            if str(roi_class) not in stat['perturbed']:
                stat['perturbed'][str(roi_class)] = list()
<<<<<<< HEAD
            stat['perturbed'][str(roi_class)].append((score, pred_boxes.tolist()[roi_idx]))
=======
            stat['perturbed'][str(roi_class)].append((score, pred_boxes.tolist()[roi_idx], float(cls_prob[roi_idx, orig_cls_pred[roi_idx]])))
>>>>>>> 8251082763a83acfdb087aca572b97621c4543f0
        
        # open('./stat-hiding-wb/{}.json'.format(p_config), 'w').write(json.dumps(stat))
        

# """""""""""""""""""""""""""""""""
#     collect refined context profile
# """""""""""""""""""""""""""""""""
        timer.tic()
        relation, rois, cls_prob, bbox_pred = sin_helper.train_eval_img(im, im_info)
        
        sorted_relation = [relation[i] for i in orders]

        # pred_boxes = pred_box_trans(rois, cls_prob, bbox_pred, im_info[-1], im_cv.shape)
        orig_cls_pred = np.argmax(cls_prob, axis=1)

        # target_context_profile = [np.empty( (256, 4096) ,dtype=float) for _ in range(5)]
        for roi_idx in range(len(orig_cls_pred)):
            roi_class = config['classes'][orig_cls_pred[roi_idx]]
            # print(roi_class)
            model = getattr(model_helper, roi_class)
            model.eval()
            context_profile = [sorted_relation[idx][roi_idx] * weights[idx] for idx in range(5)]
            context_profile = to_tensor_gpu(context_profile)
            # print(roi_class, float(model(context_profile))
            context_profile.requires_grad = True
            loss = model(context_profile)
            model.zero_grad()
            score = float(loss)
            if score > 0.09:
                loss.backward()
                data_grad = context_profile.grad.data.sign()
                context_profile = (context_profile - 0.1 * data_grad)
                new_score = float(model(context_profile))
                print 'Score changes from {} to {} for {}({})'.format(score, new_score, roi_idx, roi_class)
                if new_score < score:
                    revised_context_profile = context_profile.detach().cpu().numpy()
                    for idx in range(5):
                        sorted_relation[idx][roi_idx, :] = revised_context_profile[0,0,idx,:] / weights[idx] # cast context profile back


        
        
        target_context_profile = np.array(sorted_relation)

<<<<<<< HEAD
        # gt_boxes[box_id, -1] = t_id 
        # appear attack
        gt_boxes = np.append(gt_boxes, [[box_id[0], box_id[1], box_id[2], box_id[3], t_id]],axis = 0)
=======
        gt_boxes[box_id, -1] = t_id 
>>>>>>> 8251082763a83acfdb087aca572b97621c4543f0

        # print p_config, box_id, t_id
        # print gt_boxes
        # launch attack!
        p_im = sin_helper.train_ifgsm_attack(im, im_info, target_context_profile, gt_boxes)

<<<<<<< HEAD
        perturbation = p_im - orig_im
        perturbation = np.clip(perturbation, -10, 10)

        mask = create_mask(im_info, box_id)
        perturbation = np.multiply(perturbation, mask)

        p_im = orig_im + perturbation
        # pickle.dump(np.int32(p_im - orig_im), open('./ptrs/{}.pkd'.format(p_config),'w'))
=======
        perturbation = np.clip(p_im - orig_im, -10, 10)
        
        p_im = orig_im + perturbation

        pickle.dump(np.int32(p_im - orig_im), open('/mnt/ml/synced/SIN/step3/{}'.format(p_config),'w'))
        pickle.dump(p_im, open('join_optm.npy','w'))
>>>>>>> 8251082763a83acfdb087aca572b97621c4543f0

        print "Average time: {}".format(timer.toc())
# """""""""""""""""""""""""""""""""
#    collect stat for generated attacks in test net
# """""""""""""""""""""""""""""""""


        relation, rois, cls_prob, bbox_pred = sin_helper.eval_img(p_im, im_info)
        pred_boxes = pred_box_trans(rois, cls_prob, bbox_pred, im_info[-1], im_cv.shape)

        sorted_relation = [relation[i] for i in orders]

        cls_pred = np.argmax(cls_prob, axis=1)
        stat['join_optm'] = dict()
        for roi_idx in range(len(cls_pred)):
            roi_class = config['classes'][cls_pred[roi_idx]]
<<<<<<< HEAD
            # model = getattr(model_helper, roi_class) # white box
            model = getattr(model_helper_blackbox, roi_class) # gray box
=======
            model = getattr(model_helper, roi_class)
            # model = getattr(model_helper_blackbox, roi_class)
>>>>>>> 8251082763a83acfdb087aca572b97621c4543f0
            model.eval()
            context_profile = [sorted_relation[idx][roi_idx] * weights[idx] for idx in range(5)]
            context_profile = to_tensor_gpu(context_profile)
            # print(roi_class, float(model(context_profile))
            loss = model(context_profile)
            score = float(loss)

            if str(roi_class) not in stat['join_optm']:
                    stat['join_optm'][str(roi_class)] = list()
<<<<<<< HEAD
            stat['join_optm'][str(roi_class)].append((score, pred_boxes.tolist()[roi_idx]))


        open('./stat-appear-wb-region/{}.json'.format(p_config), 'w').write(json.dumps(stat))
=======
            stat['join_optm'][str(roi_class)].append((score, pred_boxes.tolist()[roi_idx], float(cls_prob[roi_idx, cls_pred[roi_idx]])))


        open('./stat-miscls-wb-t/{}.json'.format(p_config), 'w').write(json.dumps(stat))
>>>>>>> 8251082763a83acfdb087aca572b97621c4543f0

        # open(.format(p_config), 'w').write(json.dumps(stat))
        print '[+]{} finished!'.format(p_config)
    code.interact(local=locals())

    # while True:
    #     print(sin_helper.instance)
    #     print(sin_helper.instance2)
    #     raw_input()

if __name__ == '__main__':
    main()