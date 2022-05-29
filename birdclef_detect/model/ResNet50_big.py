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

from main_ResNet50 import Bottleneck2,Bottleneck1,Linear_dp_relu,conv_bn_relu

class Model_p(nn.Module):
    def __init__(self,dropout_rate=0):
        super().__init__()
        #input=(batch,500,128)
        number=[1,128,256,1024,2048,2048]#cnn_number

        self.cnn1=conv_bn_relu(number[0],number[1],7,stride=1,padding=5)
        self.cnn2=conv_bn_relu(number[1],number[1],3,stride=1,padding=(2,0))
        #torch.Size([10, 64, 128, 64])
        stage1=[Bottleneck1(number[1],number[2],kernel_size=3),
                Bottleneck2(number[2],int(number[2]/4),kernel_size=3),
                Bottleneck2(number[2],int(number[2]/4),kernel_size=3,groups=4)]
        self.stage1=nn.Sequential(*stage1)
        #torch.Size([10, 256, 128, 64])
        stage2=[Bottleneck1(number[2],number[3],kernel_size=3,s=2),
                Bottleneck2(number[3],int(number[3]/4),kernel_size=3,groups=4),
                Bottleneck2(number[3],int(number[3]/4),kernel_size=3),
                Bottleneck2(number[3],int(number[3]/4),kernel_size=3,groups=4)]
        self.stage2=nn.Sequential(*stage2)
        #torch.Size([10, 512, 64, 32])
        stage3=[Bottleneck1(number[3],number[4],kernel_size=3,s=2),
                Bottleneck2(number[4],int(number[4]/4),kernel_size=3,groups=4),
                Bottleneck2(number[4],int(number[4]/4),kernel_size=3),
                Bottleneck2(number[4],int(number[4]/4),kernel_size=3,groups=4),
                Bottleneck2(number[4],int(number[4]/4),kernel_size=3),
                Bottleneck2(number[4],int(number[4]/4),kernel_size=3,groups=4)]
        self.stage3=nn.Sequential(*stage3)
        #torch.Size([10, 1024, 16, 16])
        stage4=[Bottleneck1(number[4],number[5],kernel_size=3,s=2),
                Bottleneck2(number[5],int(number[5]/4),kernel_size=3,groups=4),
                Bottleneck2(number[5],int(number[5]/4),kernel_size=3),
                ]
        self.stage4=nn.Sequential(*stage4)
        #torch.Size([10, 2048, 8, 8])
        #torch.Size([10,64,2048])
        class dnn(nn.Module):
            def __init__(self):
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

        dnn_set=[dnn() for x in range(4)]
        self.dnn_set=nn.ModuleList(dnn_set)
        
        dnn_output1=Linear_dp_relu(4096,1024,dropout_rate=dropout_rate)
        dnn_output2=Linear_dp_relu(1024,152,dropout_rate=0,relu=False)
        self.dnn_final=nn.Sequential(dnn_output1,dnn_output2)
        #self.dnn_output=self.dnn_final
        class discriminate(nn.Module):
            def __init__(self):
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
            
        self.discriminate=discriminate()
        
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
        #print(x.shape)
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

############################################################################################################################################################
