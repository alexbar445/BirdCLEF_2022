import torch as tc
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import numpy as np
import torch as tc
import os,math,glob,tqdm,ujson,warnings,requests
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models.efficientnet import efficientnet_b4,efficientnet_b6

warnings.filterwarnings("ignore")

def conv_bn_relu(in_channels:int,out_channels:int,kernel_size,stride=1,padding=0,groups:int=1,relu=True,ndim=4):
    if ndim==4:
        cache=[nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,groups=groups)]
    elif ndim==3:
        cache=[nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,groups=groups)]
    else:
        print("cnn ndim error")
    cache.append(nn.BatchNorm2d(out_channels,affine=True))
    if relu:
        #cache.append(nn.LeakyReLU(1/5.5))
        #cache.append(nn.ELU())
        cache.append(nn.LeakyReLU(1e-4))
    return nn.Sequential(*cache)


class Model_p(nn.Module):
    def __init__(self,dropout_rate=0.5,):
        super().__init__()
        #input=(batch,500,128)
        #input=(batch,500,256)
        #number=[1,10,3]#cnn_number
        #self.cnn1=conv_bn_relu(number[0],number[1],5,stride=1,padding=(2,1))
        #self.cnn2=conv_bn_relu(number[1],number[2],3,stride=1,padding=(4,1))
        #torch.Size([10, 64, 128, 64])
        #self.efficientnet = tc.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b4', pretrained=True).train()
        self.efficientnet = efficientnet_b6(pretrained=True)
        
        self.dnn=nn.Linear(1000,152)
        
    def forward(self,x:tc.Tensor):
        x=x[:,None]
        
        #x=self.cnn1(x)
        #x=tc.max_pool2d(x,(2,1),padding=(0,0))
        #torch.Size([10, 10, 256, 126])
        #x=self.cnn2(x)
        """if x.shape[-1]!=128:
            d1=tc.var(x)**0.5
            cache=tc.randn(*x.shape[:-1],128,device=x.device)*d1
            index=np.random.randint(0,128-x.shape[-1])
            cache[...,index:index+x.shape[-1]]=x
            x=cache"""
        #torch.Size([10, 3, 256, 128])
        x=tc.cat([x]*3,dim=1)
        #torch.Size([10, 3, 256, 256])
        
        x=self.efficientnet(x)
        x=self.dnn(x)
        return tc.sigmoid(x)
    
if __name__ =="__main__":
    data={"batch":10*4,"skip":2,"device":"cuda:0","train_time":4000,"test_time":1000,"mode":"home","n":500,"save_time":5,
            "num_workers":5,"randn_rate":(0.6,0.2),"mix_rate":(0.7,0),"pink_noice_rate":(0.,0.),"pretrain_epoch":500,"train_epoch":100,"data_mode":"data"}
    model=Model_p().to(device=data["device"])
    x=model.forward(tc.zeros(10,500,128,device=data["device"]))
    print(x.shape)