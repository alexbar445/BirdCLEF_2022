"""https://zhuanlan.zhihu.com/p/353235794"""
import numpy as np
from pyparsing import Forward
import torch as tc
import os,math,glob,tqdm,ujson,warnings
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
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

def Linear_dp_relu(input_size,output_size,dropout_rate=0,relu=True):
    cache=[nn.Linear(input_size,output_size)]
    if relu:
        #cache.append(nn.ELU())
        #cache.append(nn.LeakyReLU(1/5.5))
        cache.append(nn.LeakyReLU(1e-4))
    if dropout_rate:
        cache.append(nn.Dropout(dropout_rate))
        #cache.append(GaussianDropout(dropout_rate))
    return nn.Sequential(*cache)

class Bottleneck1(nn.Module):#Bottleneck1
    def __init__(self,in_channels:int,out_channels:int,kernel_size,stride=1,padding=1,groups:int=1,s=False):
        super().__init__()
        self.cnn1=conv_bn_relu(in_channels,in_channels,1,stride,0,groups)
        self.cnn2=conv_bn_relu(in_channels,in_channels,kernel_size,stride,padding,groups)
        self.cnn3=conv_bn_relu(in_channels,out_channels,1,stride,0,groups,relu=False)
        self.cnn4=conv_bn_relu(in_channels,out_channels,1,stride,0,groups,relu=False)
        #self.a=tc.nn.Parameter(-tc.ones(1))
        self.s=s
    def forward(self,x:tc.Tensor):
        x1=self.cnn1.forward(x)
        x2=self.cnn4.forward(x1)
        if self.s:
            x1=tc.max_pool2d(x1,self.s)
            x2=tc.max_pool2d(x2,self.s)
        x1=self.cnn2.forward(x1)
        x1=self.cnn3.forward(x1)
        
        #a=tc.sigmoid(self.a)
        x=x1+x2
        #x=x1*a+x2*(1-a)
        return F.leaky_relu(x,1e-4)

class Bottleneck2(nn.Module):#Bottleneck2
    def __init__(self,in_channels:int,hid_channels:int,kernel_size,stride=1,padding=1,groups:int=1):
        super().__init__()
        self.cnn1=conv_bn_relu(in_channels,hid_channels,1,stride,0,groups)
        self.cnn2=conv_bn_relu(hid_channels,hid_channels,kernel_size,stride,padding,groups)
        self.cnn3=conv_bn_relu(hid_channels,in_channels,1,stride,0,groups,relu=False)
        #self.a=tc.nn.Parameter(-tc.ones(1))
    def forward(self,x:tc.Tensor):
        x1=self.cnn1.forward(x)
        x1=self.cnn2.forward(x1)
        x1=self.cnn3.forward(x1)
        
        #a=tc.sigmoid(self.a)
        x=x1+x
        #x=x1*a+x*(1-a)
        return F.leaky_relu(x,1e-4)

class dnn(nn.Module):
    def __init__(self,dropout_rate):
        super().__init__()
        self.dnn1=Linear_dp_relu(2048,512,dropout_rate=dropout_rate)
        #torch.Size([10,64,512])
        self.dnn2=Linear_dp_relu(512,128,dropout_rate=dropout_rate)
        #torch.Size([10,64,128])
        #torch.Size([10,2,4096])
        self.dnn3=Linear_dp_relu(4096,2048,dropout_rate=dropout_rate)
        #torch.Size([10,2,2048])
        #torch.Size([10,4096])
        self.dnn4=Linear_dp_relu(4096,1024,dropout_rate=dropout_rate)
        #torch.Size([10,1024])
    
    def forward(self,x:tc.Tensor):
        #torch.Size([10,64,2048])
        x=self.dnn1(x)
        #torch.Size([10,64,512])
        x=self.dnn2(x)
        #torch.Size([10,64,128])
        x=x.reshape(x.shape[0],2,-1)
        #torch.Size([10,2,4096])
        x=self.dnn3(x)
        #torch.Size([10,2,2048])
        x=x.reshape(x.shape[0],-1)
        #torch.Size([10,4096])
        x=self.dnn4(x)
        #torch.Size([10,1024])
        return x

class discriminate(nn.Module):
    def __init__(self,dropout_rate):
        super().__init__()
        self.dnn1=Linear_dp_relu(4096,1024,dropout_rate=dropout_rate)
        self.dnn2=Linear_dp_relu(1024,128,dropout_rate=dropout_rate)
        self.dnn3=Linear_dp_relu(128,1,dropout_rate=0,relu=False)
    def forward(self,x:tc.Tensor):
        #x.shape=(batch,4096)
        x=self.dnn1.forward(x)
        x=self.dnn2.forward(x)
        x=self.dnn3.forward(x)
        return x
class Model_p(nn.Module):
    def __init__(self,dropout_rate=0):
        super().__init__()
        #input=(batch,500,128)
        number=[1,64,256,512,1024,2048]#cnn_number

        self.cnn1=conv_bn_relu(number[0],number[1],7,stride=1,padding=5)
        self.cnn2=conv_bn_relu(number[1],number[1],3,stride=1,padding=2)
        #torch.Size([10, 64, 128, 64])
        stage1=[Bottleneck1(number[1],number[2],kernel_size=3),
                Bottleneck2(number[2],int(number[2]/4),kernel_size=3),
                Bottleneck2(number[2],int(number[2]/4),kernel_size=3,)]
        self.stage1=nn.Sequential(*stage1)
        #torch.Size([10, 256, 128, 64])
        stage2=[Bottleneck1(number[2],number[3],kernel_size=3,s=2),
                Bottleneck2(number[3],int(number[3]/4),kernel_size=3),
                Bottleneck2(number[3],int(number[3]/4),kernel_size=3),
                Bottleneck2(number[3],int(number[3]/4),kernel_size=3)]
        self.stage2=nn.Sequential(*stage2)
        #torch.Size([10, 512, 64, 32])
        stage3=[Bottleneck1(number[3],number[4],kernel_size=3,s=2),
                Bottleneck2(number[4],int(number[4]/4),kernel_size=3),
                Bottleneck2(number[4],int(number[4]/4),kernel_size=3),
                Bottleneck2(number[4],int(number[4]/4),kernel_size=3),
                Bottleneck2(number[4],int(number[4]/4),kernel_size=3),
                Bottleneck2(number[4],int(number[4]/4),kernel_size=3)]
        self.stage3=nn.Sequential(*stage3)
        #torch.Size([10, 1024, 16, 16])
        stage4=[Bottleneck1(number[4],number[5],kernel_size=3,s=2),
                Bottleneck2(number[5],int(number[5]/4),kernel_size=3),
                Bottleneck2(number[5],int(number[5]/4),kernel_size=3)]
        self.stage4=nn.Sequential(*stage4)
        #torch.Size([10, 2048a, 8b, 8c])
        #torch.Size([10,64b*c,2048a])
        

        dnn_set=[dnn(dropout_rate) for x in range(4)]
        self.dnn_set=nn.ModuleList(dnn_set)
        
        dnn_output1=Linear_dp_relu(4096,1024,dropout_rate=dropout_rate)
        dnn_output2=Linear_dp_relu(1024,152,dropout_rate=0,relu=False)
        self.dnn_final=nn.Sequential(dnn_output1,dnn_output2)
        #self.dnn_output=self.dnn_final
        
            
        self.discriminate=discriminate(dropout_rate)
        
    def referee_data(self):
        return self.discriminate.parameters()
    
    def feature_data(self):
        for x in [self.cnn1.parameters(),self.cnn2.parameters(),
                  self.stage1.parameters(),self.stage2.parameters(),
                  self.stage3.parameters(),self.stage4.parameters(),
                  self.dnn_set.parameters()]:
            for y in x:
                yield y
    
    def feature_forward(self,x:tc.Tensor)->tc.Tensor:
        x=x[:,None]
        
        x=self.cnn1(x)
        x=tc.max_pool2d(x,(4,2),padding=(2,0))

        x=self.cnn2(x)
        x=tc.max_pool2d(x,(2,1))

        x=self.stage1.forward(x)
        x=self.stage2.forward(x)
        x=self.stage3.forward(x)
        x=self.stage4.forward(x)
        
        x=x.reshape(x.shape[0],2048,64).transpose(1,2)
        cache=[]
        for i in range(len(self.dnn_set)):
            #x.shape=[batch,64,2048]
            cache.append(self.dnn_set[i].forward(x))
            #output.shape=[batch,1024]
        x=tc.cat(cache,dim=-1)
        #x.shape=[batch,4096]
        return x
    
    def forward(self,x:tc.Tensor,discriminate=False,feature_no_grad=False):
        """if discriminate false ->feature_no_grad=falses"""
        if discriminate:
            if feature_no_grad:
                with tc.no_grad():
                    x=self.feature_forward(x)
            else :
                x=self.feature_forward(x)
            output=self.dnn_final.forward(x)
            d=self.discriminate(x)
            return tc.sigmoid(tc.cat((output,d),dim=1))
            #output.shape=[batch,152+1]
        else :
            x=self.feature_forward(x)
            x=self.dnn_final.forward(x)
            return tc.sigmoid(x)
            #output.shape=[batch,152]

