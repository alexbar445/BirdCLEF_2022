"""https://zhuanlan.zhihu.com/p/353235794"""
import numpy as np
import torch as tc
import os,math,glob,tqdm,ujson,warnings
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
try:
    from nn_class import my_dic,Module,MyDataset,GaussianDropout,Model,train_model,test_model,set_scored_data,load_dataset,my_ELU
except:
    pass
warnings.filterwarnings("ignore")

def conv_bn_relu(in_channels:int,out_channels:int,kernel_size,stride=1,padding=0,groups:int=1,relu=True,BN=True,dropout_rate=0,ndim=4):
    if ndim==4:
        cache=[nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,groups=groups)]
    elif ndim==3:
        cache=[nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,groups=groups)]
    else:
        print("cnn ndim error")
    if BN:
        if ndim==4:
            cache.append(nn.BatchNorm2d(out_channels,affine=True))
        else:
            cache.append(nn.BatchNorm1d(out_channels,affine=True))
    if dropout_rate:
        cache.append(nn.Dropout(dropout_rate))
    if relu:
        #cache.append(nn.LeakyReLU(1/5.5))
        #cache.append(nn.GELU())
        cache.append(my_ELU())
        #cache.append(nn.LeakyReLU(1e-4))
    return nn.Sequential(*cache)

def Linear_dp_relu(input_size,output_size,dropout_rate=0,relu=True):
    cache=[nn.Linear(input_size,output_size)]
    if relu:
        #cache.append(nn.ELU())
        #cache.append(nn.LeakyReLU(1/5.5))
        #cache.append(nn.LeakyReLU(1e-4))
        cache.append(my_ELU())
        #cache.append(nn.GELU())
    if dropout_rate:
        cache.append(nn.Dropout(dropout_rate))
        #cache.append(GaussianDropout(dropout_rate))
    return nn.Sequential(*cache)

def stem(in_channels:int,op_channels:int,dim:list,kernel_size=4,stride=2,padding=1,groups:int=1):
    return nn.Sequential(nn.Conv2d(in_channels, op_channels, kernel_size=kernel_size, stride=stride,padding=padding,groups=groups),
                        nn.LayerNorm(dim, eps=1e-6,)
                    )
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

class Block(nn.Module):#Bottleneck2
    def __init__(self,in_channels:int,hid_channels:int,dim:list,kernel_size=7,stride=1,padding=3,groups:int=32):
        super().__init__()
        self.cnn1=nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,groups=groups)
        self.LN=nn.LayerNorm(dim)
        self.cnn2=nn.Conv2d(in_channels,hid_channels,1,stride,0,groups=1)
        #self.act=nn.LeakyReLU(1e-3)
        self.act=my_ELU()
        #self.act=nn.GELU()
        self.cnn3=nn.Conv2d(hid_channels,in_channels,1,stride,0,groups=1)
    def forward(self,x:tc.Tensor):
        x1=self.cnn1.forward(x)
        x1=self.LN.forward(x1)
        x1=self.cnn2.forward(x1)
        x1=self.act.forward(x1)
        x1=self.cnn3.forward(x1)
        
        x=x1+x
        return x
        
class Model_p(nn.Module):
    def __init__(self,dropout_rate=0):
        super().__init__()
        number=[1,48,96,192,384,768]#cnn_number
        #input=(batch,500,124)
        self.cnn1=conv_bn_relu(number[0],number[1],7,stride=1,padding=(4,5))
        self.cnn2=conv_bn_relu(number[1],number[2],5,stride=1,padding=(2,2))
        """self.cnn1=nn.Sequential(nn.Conv2d(number[0],number[2],7,stride=(4,2),padding=(5,5)),
                                nn.LayerNorm([number[2],126,64]),
                                nn.LeakyReLU(1e-3))
        #input=(batch,72,128)
        self.cnn2=nn.Sequential(nn.Conv2d(number[2],number[2],7,stride=(2,1),padding=(4,3)),
                                nn.LayerNorm([number[2],64,64]),
                                nn.LeakyReLU(1e-3))"""
        #torch.Size([10, 64, 64, 64])
        stage1=[]
        for x in range(3):
            stage1.append(Block(number[2],number[2]*4,dim=[number[2],64,64],kernel_size=7,stride=1,padding=3,groups=32))
        self.stage1=nn.Sequential(*stage1)
        #torch.Size([10, 256, 64, 64])
        
        #stage2=[stem(number[2],number[3],dim=[number[3],32,32],stride=2)]
        stage2=[Bottleneck1(number[2],number[3],kernel_size=3,s=2)]
        for x in range(3):
            stage2.append(Block(number[3],number[3]*4,dim=[number[3],32,32],kernel_size=7,stride=1,padding=3,groups=32))
        self.stage2=nn.Sequential(*stage2)
        #torch.Size([10, 512, 32, 32])
        stage3=[Bottleneck1(number[3],number[4],kernel_size=3,s=2)]
        #stage3=[stem(number[3],number[4],dim=[number[4],16,16],stride=2)]
        for x in range(9):
            stage3.append(Block(number[4],number[4]*4,dim=[number[4],16,16],kernel_size=7,stride=1,padding=3,groups=32))
        self.stage3=nn.Sequential(*stage3)
        #torch.Size([10, 1024, 16, 16])
        stage4=[Bottleneck1(number[4],number[5],kernel_size=3,s=2)]
        #stage4=[stem(number[4],number[5],dim=[number[5],8,8],stride=2)]
        for x in range(3):
            stage4.append(Block(number[5],number[5]*4,dim=[number[5],8,8],kernel_size=7,stride=1,padding=3,groups=32))
        self.stage4=nn.Sequential(*stage4)
        #torch.Size([10, 2048, 8, 8])
        
        #torch.Size([10,64,2048])
        class dnn(nn.Module):
            def __init__(self,input_shape=2048):
                super().__init__()
                self.dnn1=Linear_dp_relu(input_shape,input_shape//2,dropout_rate=dropout_rate)
                #torch.Size([10,64,512])
                self.dnn2=Linear_dp_relu(input_shape//2,128,dropout_rate=dropout_rate)
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

        dnn_set=[dnn(768) for x in range(4)]
        self.dnn_set=nn.ModuleList(dnn_set)
        
        dnn_output1=Linear_dp_relu(4096,1024,dropout_rate=dropout_rate)
        dnn_output2=Linear_dp_relu(1024,152,dropout_rate=0,relu=False)
        self.dnn_final=nn.Sequential(dnn_output1,dnn_output2)
        
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
        x=tc.max_pool2d(x,(4,2),padding=(1,0))
        
        x=self.cnn2(x)
        x=tc.max_pool2d(x,(2,1),padding=(1,0))
        
        x=self.stage1.forward(x)
        x=self.stage2.forward(x)
        x=self.stage3.forward(x)
        x=self.stage4.forward(x)
        x=x.permute(0,2,3,1)
        x=x.reshape(x.shape[0],64,-1)
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

