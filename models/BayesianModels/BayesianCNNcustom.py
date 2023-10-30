import torch.nn as nn
import math
from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper
import config_bayesian as cfg

class BBBCustom(ModuleWrapper):
    '''Custom network with Bayesian Layers'''
    '''For now: Stucture is the same as AlexNet, but with different input size for images'''

    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus', imgsize=None):
        super(BBBCustom, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors
        if imgsize == None:
            self.imgsize = cfg.imgsize
        else:
            self.imgsize = imgsize

        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")
        

        """
        Custom network with Bayesian Layers for POCUS dataset
        Network structure is based on last publication for the POCUS dataset
        """        

        dropout = False

        self.conv1 = BBBConv2d(inputs, 32, 3, bias=True, priors=self.priors)
        self.act1 = self.act()
        if dropout: self.drop1 = nn.Dropout2d(0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(32, 64, 3, bias=True, priors=self.priors)
        self.act2 = self.act()
        if dropout: self.drop2 = nn.Dropout2d(0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BBBConv2d(64, 128, 3, bias=True, priors=self.priors)
        self.act3 = self.act()
        if dropout: self.drop3 = nn.Dropout2d(0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = BBBConv2d(128, 128, 3, bias=True, priors=self.priors)
        self.act4 = self.act()
        if dropout: self.drop4 = nn.Dropout2d(0.2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.conv5 = BBBConv2d(128, 128, 3, bias=True, priors=self.priors)
        self.act5 = self.act()
        if dropout: self.drop5 = nn.Dropout2d(0.2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        if dropout: self.drop6 = nn.Dropout2d(0.5)

        if self.imgsize == 180:
            n_flat = 128*3*3
        elif self.imgsize == 128:
            n_flat = 512
        else:
            raise ValueError("Only imgsize 180 or 128 supported")

        self.flatten = FlattenLayer(n_flat) # 128*3*3 BEFORE (for images 180x180). 512 for images 128x128
 
        
        self.fc1 = BBBLinear(n_flat, 512, bias=True, priors=self.priors)
        self.act6 = self.act()
        if dropout: self.drop7 = nn.Dropout2d(0.5)

        self.fc2 = BBBLinear(512, outputs, bias=True, priors=self.priors)



