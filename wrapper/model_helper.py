import sys
import os
# sys.path.append('../attack_detector')

# sys.path.append('../lib')

import yaml
import torch

with open("config.yml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
        # print(config)
    except yaml.YAMLError as exc:
        print 'YAML parse error:', exc

for lib_path in config['sys_path']:
    if lib_path not in sys.path:
        sys.path.append(lib_path)

from auto_encoder import AutoEncoder

class ModelHelper:
    """
        This helper is going to help you get the model
        and load them when needed.

        The path configuration is from config.yml in 
        SCEME section
    """
    def __init__(self,version=None, c=None):
        if c is not None:
            self.config = c
        else:
            self.config = config
        self.models = {}
        self.use_gpu = torch.cuda.device_count()>0
        if version is not None:
            self.version = version
        else:
            self.version = self.config['SCEME']['checkpoint']['version']

    def __getitem__(self, name):

        if name in self.models:
            return self.models[name]
        else:
            model = AutoEncoder( self.config['SCEME']['gamma'])
            if self.use_gpu:
                model = torch.nn.DataParallel(model).cuda()

            model_filename = '_'.join([ self.config['SCEME']['checkpoint']['name'], self.version, 'epoch' +  self.config['SCEME']['checkpoint']['epoch'] ]) + '.pt'
            # print(model_filename)
            checkpoint_name = os.path.join( self.config['SCEME']['path'],  self.config['SCEME']['prefix'] + name, model_filename)
            # print(checkpoint_name)
            print 'load model ' + checkpoint_name

            model.load_state_dict(torch.load(checkpoint_name))
            self.models[name] = model
            return model

    def __getattr__(self, name):
        return self.__getitem__(name)

    @staticmethod
    def fgsm_attack(model, context_profile):
        def calc_perturbed_profile(original, data_grad, epsilon = 0.1):
            # Collect the element-wise sign of the data gradient
            sign_data_grad = data_grad.sign()
            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_context_profile = original - epsilon*sign_data_grad
            # Return the perturbed profile
            return perturbed_context_profile

        # enable evaluation mode        
        model.eval()

        data = context_profile
        data.requires_grad = True
        loss = model(data)

        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        perturbed_data = calc_perturbed_profile(data, data_grad) # epsilon by default eqauls 0.1

        output = model(perturbed_data)

        return perturbed_data, float(output)

if __name__ == '__main__':
    a = ModelHelper()
    # a.b
    a['aeroplane']
    for target_cls in config['classes']:
        a[target_cls]
    print('Load finished!')
    for target_cls in config['classes']:
        print(a[target_cls])

    print('All load success!')
    # while True:
    #     try:
    #         target_cls = raw_input()
    #         print( a[target_cls] )
    #     except KeyboardInterrupt:
    #         break
    # a._print('hello')