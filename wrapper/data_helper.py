import sys
import tensorflow as tf
import yaml

import os
__filepath__ = os.path.dirname(os.path.realpath(__file__)) + '/'

with open(__filepath__ + "config.yml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
        # print(config)
    except yaml.YAMLError as exc:
        print 'YAML parse error:', exc

for lib_path in config['sys_path']:
    if lib_path not in sys.path:
        sys.path.append(__filepath__ + lib_path)

# ../attack_detector
from attack_aux import prepare_dataset

# ../context_profile
from test_AE_aux import get_image_prepared

# ../lib/fast_rcnn
from fast_rcnn.config import cfg, cfg_from_file

cfg_from_file(config['fast_rcnn']['config_path'])

class DataHelper:
    class __DataHelper:
        def __init__(self, dataset_name, cfg, data_list):
            self.cfg = cfg
            self.dataset = dataset_name
            self.imdb = prepare_dataset(dataset_name, cfg)
            if data_list is None:
                data_list = config['data']['list_path']
            with open(data_list, 'r') as f:
                self.list = [int(idx.strip().split('_')[-1]) for idx in list(f)]

        def get_image(self,idx):
            """
                @parameter idx image index
                @return im_cv, im, im_info, gt_boxes
            """
            return get_image_prepared(self.cfg, self.imdb.roidb[idx])
        
        def get_image_by_name(self,name):
            for enum, idx in enumerate(self.list):
                if int(idx) == int(name):
                    return self.get_image(enum)
    
    instance = None
    global cfg
    def __init__(self, dataset_name = 'voc_2007_test', cfg=cfg, data_list=None):
        if DataHelper.instance is None:
            DataHelper.instance = DataHelper.__DataHelper(dataset_name, cfg, data_list)
    
    def __getattr__(self, name):
        return getattr(DataHelper.instance, name)


if __name__ == '__main__':
    dh = DataHelper('voc_2007_test',cfg)
    print dh.get_image_by_name(313)