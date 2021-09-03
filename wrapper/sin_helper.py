import sys, os, shutil, cv2, pickle
import numpy as np
import tensorflow as tf
import yaml

with open("config.yml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
        # print(config)
    except yaml.YAMLError as exc:
        print 'YAML parse error:', exc

for lib_path in config['sys_path']:
    if lib_path not in sys.path:
        sys.path.append(lib_path)

# ../lib/networks
from networks.factory import get_network

# ../lib/fast_rcnn

from fast_rcnn.config import cfg, cfg_from_file

cfg_from_file(config['fast_rcnn']['config_path'])

# ../context_profile
from train_aux import get_rpn_cls_loss, get_rpn_box_loss
from train_aux import get_RCNN_cls_loss, get_RCNN_box_loss


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_context_profile_loss(net):
	model_context_profile = net.relation
	target_context_profile = 0
    # target_context_profile = np.zeros([1])

	node_feature = model_context_profile[0]
	reset_objects = model_context_profile[3]
	reset_scene = model_context_profile[1]
	update_objects = model_context_profile[4]
	update_scene = model_context_profile[2]

	context_profile = tf.stack([node_feature, reset_objects, reset_scene, update_objects, update_scene], 0)

	loss = tf.reduce_sum(tf.abs(tf.subtract(context_profile ,target_context_profile)), reduction_indices=[1])

	context_profile_loss = tf.reduce_mean(loss)

	return context_profile_loss


SIN_MUTEX = None

class SinHelper:
    class __SinHelper:
        def __init__(self, ckpt_path = None):
            self.g = tf.Graph()
            with self.g.as_default():
                with HiddenPrints():
                    self.net = get_network("VGGnet_train")
                self.saver = tf.train.Saver()

                rpn_cls_loss = get_rpn_cls_loss(self.net)
                rpn_box_loss = get_rpn_box_loss(self.net)
                RCNN_cls_loss = get_RCNN_cls_loss(self.net)
                RCNN_box_loss = get_RCNN_box_loss(self.net)
                context_profile_loss = get_context_profile_loss(self.net)
                # !!!! here optimizer not miminize loss, the gradient is to maximum loss!
                # every sign should be opposite!!
                loss = rpn_cls_loss + rpn_box_loss + RCNN_box_loss + context_profile_loss

                grad, = tf.gradients(-loss, self.net.data)  # default maximize
                self.grad = tf.sign(grad)

                self.fetch_list = [self.net.relation,
                                   self.net.get_output('rpn_rois'),
                                   self.net.get_output('cls_prob'),
                                   self.net.get_output('bbox_pred')
                                ]
            tf_config = tf.ConfigProto(allow_soft_placement=True)
            tf_config.gpu_options.allow_growth = True

            net_filename = config['SIN']['ckpt'] if ckpt_path is None else ckpt_path
            self.sess = tf.Session(config=tf_config, graph=self.g)
            self.saver.restore(self.sess, net_filename)

            print('Loading model weights from {:s}'.format(net_filename))
        
        def ifgsm_attack(self, img, im_info, target_relation, gt_boxes, max_iteration = 10):
            # iteration = 0
            
            # # while iteration < max_iteration:
            #     iteration += 1
            feed_dict = {
                self.net.data: np.expand_dims(img, axis=0),
                self.net.im_info: np.expand_dims(im_info, axis=0),
                self.net.gt_boxes: gt_boxes,
                self.net.appearance_drop_rate: 0.0,
                self.net.target_relation: target_relation
            }
            cur_grad = self.sess.run(self.grad, feed_dict = feed_dict)
            p = np.squeeze(cur_grad)
            p_im = np.clip(img + p + cfg.PIXEL_MEANS, 0, 255) - cfg.PIXEL_MEANS
            return p_im

        def eval_img(self, img, im_info):
            feed_dict = {
                self.net.data: np.expand_dims(img, axis = 0),
                self.net.im_info: np.expand_dims(im_info, axis = 0),
                self.net.appearance_drop_rate: 0.0,
                self.net.gt_boxes: np.expand_dims([0,0,0,0,0], axis = 0),
                self.net.target_relation: np.zeros((1))

            }

            return self.sess.run(self.fetch_list, feed_dict=feed_dict)


    class __SinHelper2:
        def __init__(self, ckpt_path = None):
            self.g = tf.Graph()
            with self.g.as_default():
                self.net = get_network("VGGnet_test")
                self.saver = tf.train.Saver()
                self.fetch_list = [self.net.relation,
                                   self.net.get_output('rois'),
                                   self.net.get_output('cls_prob'),
                                   self.net.get_output('bbox_pred')
                                ]

            tf_config = tf.ConfigProto(allow_soft_placement=True)
            tf_config.gpu_options.allow_growth = True

            net_filename = config['SIN']['ckpt'] if ckpt_path is None else ckpt_path
            self.sess = tf.Session(config=tf_config, graph=self.g)
            self.saver.restore(self.sess, net_filename)

            print('Loading model weights from {:s}'.format(net_filename))
        
        def eval_img(self, img, im_info):
            feed_dict = {
                self.net.data: np.expand_dims(img, axis = 0),
                self.net.im_info: np.expand_dims(im_info, axis = 0),
                self.net.appearance_drop_rate: 0.0
            }

            return self.sess.run(self.fetch_list, feed_dict=feed_dict)


    # Only initialize once
    instance = None
    instance2 = None
    def __init__(self):
        if not SinHelper.instance:
            SinHelper.instance = SinHelper.__SinHelper()
        # print('train model loaded success!')
        # raw_input()
        if not SinHelper.instance2:
            SinHelper.instance2 = SinHelper.__SinHelper2(ckpt_path = config['SIN']['surrogate_ckpt'])

    def __getattr__(self, name):
        if name[:5] == "train":
            return getattr(self.instance, name[6:])
        else:
            return getattr(self.instance2, name.replace('test_', ''))

if __name__ == '__main__':
    sin = SinHelper()

    print('all load success!')