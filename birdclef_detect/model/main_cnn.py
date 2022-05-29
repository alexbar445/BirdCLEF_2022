import numpy as np
import torch as tc
import os,math,glob,tqdm,ujson
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from nn_class import my_dic,MyDataset
import warnings
warnings.filterwarnings("ignore")

def conv_bn_relu(in_ch,op_ch,kernel_size,stride,padding,groups=1):
            return nn.Sequential(nn.Conv2d(in_ch,op_ch,kernel_size,stride=stride,padding=padding,groups=groups),
                                nn.BatchNorm2d(op_ch,affine=True),
                                nn.LeakyReLU(1e-4))
            
class Model_p(nn.Module):
    def __init__(self,dropout_rate=0):
        super().__init__()
        number=[1,64,128,256]#cnn_number
        #number=[1,8,16,32]#cnn_number
        self.save_x=0
        #input=(batch,500,120)
        def conv_bn_relu(in_ch,op_ch,kernel_size,stride,padding,groups=1):
            return nn.Sequential(nn.Conv2d(in_ch,op_ch,kernel_size,stride=stride,padding=padding,groups=groups),
                                nn.BatchNorm2d(op_ch,affine=True),
                                nn.LeakyReLU(1e-4))
        
        self.cnn1=conv_bn_relu(number[0],number[1],(7,5),stride=1,padding=(3,0))#x=(batch,n,500,116)
        self.cnn2=conv_bn_relu(number[1],number[2],3,stride=1,padding=(2,2))#x=(batch.n,102,39)
        self.cnn3=conv_bn_relu(number[2],number[3],3,stride=1,padding=(2,2))#x=(batch.n,36,21)
        self.cnn4=conv_bn_relu(number[3],number[3],3,stride=1,padding=1,groups=8)#x=(batch.n,12,12)
        self.cnn5=conv_bn_relu(number[3],number[3],3,stride=1,padding=1,groups=8)#x=(batch.n,12,12)
        self.cnn6=conv_bn_relu(number[3],number[3],3,stride=1,padding=1,groups=8)#x=(batch.n,12,12)
        self.cnn7=conv_bn_relu(number[3],number[3],3,stride=1,padding=1,groups=8)#x=(batch.n,12,12)
        self.cnn8=conv_bn_relu(number[3],number[3],3,stride=1,padding=1,groups=8)#x=(batch.n,12,12)
        self.maxpool=nn.MaxPool2d(kernel_size=2)
        
        self.dnn1=nn.Linear(number[3]*6*6,4096)
        self.dnn2=nn.Linear(4096,2048)
        self.dnn3=nn.Linear(2048,1024)
        self.dnn4=nn.Linear(1024,152)
        self.dp=nn.Dropout(dropout_rate)
        
    def forward(self,x:tc.Tensor):
        x=x[:,None]

        x=self.cnn1(x)
        x=tc.max_pool2d(x,(5,3))
        x=self.cnn2(x)
        x=tc.max_pool2d(x,(3,2))
        x=self.cnn3(x)
        x=tc.max_pool2d(x,(3,2),padding=(0,1))
        x=self.cnn4(x)
        x=self.cnn5(x)
        x=self.cnn6(x)
        x=self.cnn7(x)
        x=self.cnn8(x)
        x=self.maxpool(x)
        x=x.reshape(x.shape[0],-1)
        ##############################
        x=self.dnn1(x)
        x=F.leaky_relu(x,1e-4)
        x=self.dp(x)
        
        x=self.dnn2(x)
        x=F.leaky_relu(x,1e-4)
        x=self.dp(x)
        
        x=self.dnn3(x)
        x=F.leaky_relu(x,1e-4)
        x=self.dp(x)
        
        x=self.dnn4(x)
        return x

class Model(nn.Module):
    def __init__(self,dropout_rate=0.1,
                 optimizer=tc.optim.AdamW,lr=1e-4,betas=(0.9,0.999),weight_decay=0.03,
                 loss_fc:nn.BCELoss=nn.BCELoss(reduction="sum")):
        super().__init__()
        self.cnn=Model_p(dropout_rate)
        cache=dict(lr=1e-4,betas=(0.9,0.999),weight_decay=0.03)
        self.optimizer=optimizer(self.parameters(),**cache)
        self.loss_fc=loss_fc
        
    def forward(self,x:tc.Tensor):
        #index is record the park of batch 
        #index.shape=(batch,n)
        #x.shape=(batch,2000,257)
        #model output shape=(batch*4,157)
        #use index to reshape to (batch,157)
        #output=tc.nan_to_num(output,0.0)
        #output=(output-output.min())/(output.max()-output.min())

        cache=x.reshape(x.shape[0],4,500,-1)
        max_=cache.max(dim=3)[0].max(dim=2)[0][:,:,None,None]
        min_=cache.min(dim=3)[0].min(dim=2)[0][:,:,None,None]
        cache=(min_-cache)/(min_-max_)
        output=self.cnn(cache.reshape(x.shape[0]*4,500,-1))
        del cache
        output=output.reshape(x.shape[0],4,output.shape[-1])
        
        output=output.max(dim=1)[0]
        output=tc.sigmoid(output)
        
        #output=tc.sigmoid(output.sum(1))
        return output
    
    def auto_train(self,x:tc.Tensor,y:tc.Tensor,mask=None):
        optimizer.zero_grad()

        if x.shape[1]==500:
            output=self.cnn(x)
            output=tc.sigmoid(output)
        else :
            output=self.forward(x)
        if not (mask is None):
            output=output[:,mask]
            
        loss=loss_fc.forward(output,y)/output.shape[0]
        loss.backward()
        optimizer.step()
        return {"output":output,"loss":loss}
    
    def test(self,x:tc.Tensor):
        #x.shape=(batch,500,257)
        output=self.cnn(x)
        output=output>=0
        return output
       
    def test_error(self,x:tc.Tensor):
        cache=x.reshape(x.shape[0],4,500,-1)
        max_=cache.max(dim=3)[0].max(dim=2)[0][:,:,None,None]
        min_=cache.min(dim=3)[0].min(dim=2)[0][:,:,None,None]
        cache=(min_-cache)/(min_-max_)
        output=self.cnn(cache.reshape(x.shape[0]*4,500,-1))
        #del cache
        output1=output.reshape(x.shape[0],4,output.shape[-1])
        output=output1.max(dim=1)[0]
        output=tc.sigmoid(output)

        return output,output1,cache.reshape(x.shape[0],500*4,-1)
      
    def load_checkpoint(self, checkpoint, optimizer=None):
        if checkpoint != 'No':
            print("loading checkpoint...")
            model_dict = self.state_dict()
            modelCheckpoint = tc.load(checkpoint)
            pretrained_dict = modelCheckpoint['state_dict']
            # 过滤操作
            new_dict={}
            for k,v in pretrained_dict.items():
                if k in model_dict.keys() and model_dict[k].shape==v.shape:
                    #print(k,v.shape)
                    new_dict[k]=v
                else :
                    try:
                        print(f"pass: {k} data_v: {v.shape} model_v: {model_dict[k].shape}")
                    except:
                        print(f"pass: {k} data_v: {v.shape} model_v: None")
                    input("enter?")
                    v:tc.Tensor
                    #cache=v.clone()
                    model_dict[k]*=0
                    cache=model_dict[k].shape
                    try:
                        cache=v.shape
                        if v.ndim==4:
                            model_dict[k][:cache[0],:cache[1],:cache[2],:cache[3]]+=v
                        elif v.ndim==3:
                            model_dict[k][:cache[0],:cache[1],:cache[2]]+=v
                        elif v.ndim==2:
                            model_dict[k][:cache[0],:cache[1]]+=v
                        elif v.ndim==1:
                            model_dict[k][:cache[0]]+=v
                        else:
                            break
                        print("get finish")
                    except:
                        #v=cache
                        print("get error")
                    optimizer=tc.optim.AdamW(self.parameters(),lr=1e-4,betas=(0.9,0.9))
                    
            #new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
            self.load_state_dict(model_dict)
            print("loaded finished!")
            # 如果不需要更新优化器那么设置为false
            #optimizer.load_state_dict(modelCheckpoint['optimizer'])
            try:
                optimizer.load_state_dict(modelCheckpoint['optimizer'])
            except:
                print('not loaded optimizer')
        else:
            print('No checkpoint is included')
        print("load checkpoint Succeeded")

    def save(self,optimizer:tc.optim.AdamW,path=None,loss=None):
        try:
            cache=self.save_x
        except:
            self.save_x=0 
        if path:
            tc.save({"state_dict":self.state_dict(),"optimizer":optimizer.state_dict()},path+"_%2f.pth"%loss)
        else:
            if "model_data" not in glob.glob("*"):
                os.system("mkdir model_data")
            if loss==None:
                tc.save({"data":self.save_x,"state_dict":self.state_dict(),"optimizer":optimizer.state_dict()
                         },"model_data/"+str(self.save_x)+".pth")
            else :
                tc.save({"data":self.save_x,"state_dict":self.state_dict(),"optimizer":optimizer.state_dict()
                         },"model_data/"+str(self.save_x)+"_%2f.pth"%loss)
                
            self.save_x+=1

############################################################################################################################################################
if __name__ =="__main__":
    tc.backends.cudnn.benchmark = True###new
    data={"batch":10,"skip":2,"device":"cuda:0","train_time":1000,"test_time":1000,"mode":"home","n":500,"num_workers":2,"randn_rate":0.2}
    data["train_time"]=8000
    data["test_time"]=1000
    data["train_time_m"]=1
    if data["mode"]=="home":
        data["model_path"]="model_data/0_151.858359.pth"
        data["dc_path"]="D:/python_data/voice/birdclef_voice/voice_data/"
        data["train_path"]="D:/python_data/voice/birdclef_voice/voice_data/train_data/"
        data["test_path"]="D:/python_data/voice/birdclef_voice/voice_data/test_data/"
        data["num_workers"]=0
    elif data["mode"]=="out":
        data["model_path"]="model_data/main.pth"
        data["dc_path"]="./"
        data["train_path"]="train_data/"
        data["test_path"]="test_data/"
        data["num_workers"]=5
        data["batch"]=30
    else: 
        data["model_path"]="../input/cache-data/main2.pth"
        data["dc_path"]="../input/data123/"
        data["train_path"]="../input/train-data/train_data/"
        data["test_path"]="../input/test-data/test_data/"
        data["num_workers"]=2



    
    model=Model(0.4)

    model=model.to(data["device"])
    ####################################################################################################################################
    optimizer=tc.optim.AdamW(model.parameters(),lr=1e-4,betas=(0.9,0.9),weight_decay=0.01)
    try:
        model.load_checkpoint(data["model_path"],optimizer=optimizer)
        """cache=model.cnn.state_dict()
        for k in cache.keys():
            if k[:3]=="cnn"or k[:5]=="stage":
                print(k,"requires_grad=False")
                cache[k].requires_grad=True"""
    except:
        print("load error")
    #optimizer=tc.optim.AdamW(model.parameters(),lr=1e-4,betas=(0.9,0.9))
            
    print("old_learn_rate:",optimizer.param_groups[0]['lr'])
    optimizer.param_groups[0]['lr'] = 1e-4
    print("updata: ",optimizer.param_groups[0]['lr'])
    
    ####################################################################################################################################
    dc=my_dic()
    try:
        dc.load(data["dc_path"])
    except:
        with open(data["dc_path"]+"data.json","r") as file:
            y_train=ujson.load(file)["y"]
        for x in y_train:
            for t in x:
                dc.ck(t) 
        dc.data.sort()
        dc.save(data["dc_path"])
    #print(len(dc.data))
    
    try:
        with open(data["train_path"]+"x_more.json","r") as file:
            paths=ujson.load(file)
            paths=np.array(paths,dtype=np.object0)
        labels=tc.load(data["train_path"]+"y_more.pt").to(tc.bool)
        train_dataset=MyDataset(paths,labels,dc,data["train_path"],reset=True,n=data["n"],randn=data["randn_rate"])
        
        with open(data["test_path"]+"x_data.json","r") as file:
            paths=ujson.load(file)
            paths=np.array(paths,dtype=np.object0)
        labels=tc.load(data["test_path"]+"y_data.pt").to(tc.bool)
        test_dataset=MyDataset(paths,labels,dc,data["test_path"],reset=True,n=data["n"],randn=0.01)
    except FileNotFoundError:
        #print("error")
        #exit()
        path="D:/python_data/voice/birdclef_voice/voice_data/data/"
        with open(path+"data.json","r") as file:
            cache=ujson.load(file)
        cache=MyDataset(cache["x"],cache["y"],dc,path,reset=True)
        cache.cut(path=path,rate=0.1,
                  train_data_path="D:/python_data/voice/birdclef_voice/voice_data/train_data/",
                  test_data_path="D:/python_data/voice/birdclef_voice/voice_data/test_data/")
        exit()
        from Data_Augmentation import data_augmentation
        data_augmentation(path+"train_data/",300)
        data_augmentation(path+"test_data/",20)
        
        with open(path+"train_data/x_more.json","r") as file:
            paths=ujson.load(file)
            paths=np.array(paths,dtype=np.object0)
        labels=tc.load(path+"train_data/y_more.pt").to(tc.bool)
        train_dataset=MyDataset(paths,labels,dc,path+"train_data/",reset=True)
        
        with open(path+"test_data/x_more.json","r") as file:
            paths=ujson.load(file)
            paths=np.array(paths,dtype=np.object0)
        labels=tc.load(path+"test_data/y_more.pt").to(tc.bool)
        test_dataset=MyDataset(paths,labels,dc,path+"test_data/",reset=True)
        
    #filer_data
    if False:
        with open("D:/python_data/voice/birdclef_voice/scored_birds.json","r")as file:
            target=ujson.load(file)
        data["scored"]=dc.transform(target).sum(0).to(tc.bool)
        del target
        index=train_dataset.labels[:,data["scored"]]
        train_dataset.paths=train_dataset.paths[index.sum(1).to(tc.bool)]
        train_dataset.labels=train_dataset.labels[index.sum(1).to(tc.bool)][:,data["scored"]]
        
        index=test_dataset.labels[:,data["scored"]]
        test_dataset.paths=test_dataset.paths[index.sum(1).to(tc.bool)]
        test_dataset.labels=test_dataset.labels[index.sum(1).to(tc.bool)][:,data["scored"]]
        """def run1(self,x:tc.Tensor):
            #x.shape=(batch,500,257)
            output=self.cnn(x)
            output=tc.sigmoid(output)[:,data["scored"]]
            return output
        model.forward=run1
        cache=model.cnn.state_dict()
        for k in cache.keys():
            if k[:3]!="cnn":
                cache[k].requires_grad=False"""

    #train_dataset.calculate()
    #test_dataset.calculate()
    
    a,Recall,Precision=0,0,0
    loss_fc=nn.BCELoss(reduction="sum")

    cache={"a":[],"point":[],"test":[]}
    
    train_dataset.re_set(data["train_time"],repeat=data["train_time_m"],mode=True,add_empty=True)
    test_dataset.re_set(data["test_time"],mode=True,add_empty=True)
    
    train_dataset = tc.utils.data.DataLoader(train_dataset, batch_size=data["batch"],shuffle=True,num_workers=data["num_workers"])#構建tc資料集
    test_dataset = tc.utils.data.DataLoader(test_dataset, batch_size=data["batch"],shuffle=True,num_workers=data["num_workers"])#構建tc資料集
    
    
    def count_point(output:tc.Tensor,targer:tc.Tensor)-> float:
        """F1 point"""
        Recall,Precision=0,0
        for x in range(output.shape[0]):
            cache1=(output[x]>0.5)
            cache2=(cache1[targer[x].to(tc.bool)]).to(tc.float32)#Recall
            cache3=(y_train[x,cache1.to(tc.bool)]).to(tc.float32)#Precision
            if cache2.numel()!=0:#expty targer
                Recall+=cache2.mean().cpu()/output.shape[0]
            else :
                Recall+=1/output.shape[0]
            if cache3.numel()!=0:
                Precision+=cache3.mean().cpu()/output.shape[0]
            else :
                Precision+=1/output.shape[0]
            
            F1=(2*Precision*Recall)/(Precision+Recall)
            if Precision+Recall==0:
                return 0
            else :
                if isinstance(F1,tc.Tensor):
                    return F1.tolist()
                return F1

    for epoch in range(20000):
        print("\n")

        model=model.train()
        if epoch !=0:
            train_dataset.dataset.re_set(data["train_time"],repeat=data["train_time_m"],add_empty=False,mode=True)
            test_dataset.dataset.re_set(data["test_time"],add_empty=False,mode=True)
        
        i=1
        time=tqdm.tqdm(total=len(train_dataset.dataset),desc="hi")
        for x_train,y_train in train_dataset:
            i+=1
            x_train=x_train.to(data["device"])
            y_train=y_train.to(data["device"])
            
            if data["n"]==2000:
                index=np.random.randint(0,x_train.shape[1])
                x_train=tc.cat((x_train[:,index:],x_train[:,:index]),dim=1)#Data_Augmentation
            
            if y_train.shape[1]==21:
                output=model.auto_train(x_train,y_train,mask=data["scored"])
            else:
                output=model.auto_train(x_train,y_train)
            
            loss=output["loss"]
            output=output["output"]
            
            cache["point"].append(count_point(output,y_train))
            
            cache["a"].append(loss.sum().cpu().tolist())
            
            if i%data["skip"]==0:
                time.set_description("loss: %.3f , point: %.3f"%(np.array(cache['a']).mean(),
                                                            np.array(cache['point']).mean()))
                time.update(data["skip"]*data["batch"])
            
        model.save(optimizer,loss=np.array(cache["a"]).mean())
        del y_train,x_train
        
        model=model.train(False)
        
        time=tqdm.tqdm(total=len(test_dataset.dataset),desc="hi")
        loss=[]
        print("/n")
        i=1
        with tc.no_grad():
            for x_test,y_test in test_dataset:
                i+=1
                x_test=x_test.to(data["device"])
                y_test=y_test.to(data["device"])
                if data["n"]==500:
                    output=model.cnn.forward(x_test)
                    output=tc.sigmoid(output)
                else :
                    output=model.forward(x_test)
                
                if y_test.shape[1]==21:
                    output=output[:,data["scored"]]
                    
                cache["test"].append(count_point(output,y_test))
                    
                if i%data["skip"]==0:
                    time.set_description("test->loss: %.3f , point: %.3f"%(np.array(loss).mean(),
                                                                np.array(cache["test"]).mean()))
                    time.update(data["skip"]*data["batch"])
                    
        del x_test,y_test
        
        
        
        try :
            with open("data.json" ,"r") as file:
                cache_h=ujson.load(file)
        except FileNotFoundError:
            cache_h={"a":[],"point":[],"test":[]}

        with open("data.json" ,"w") as file:
            cache_h["test"]+=np.array(cache["test"]).tolist()
            cache_h["a"]+=np.array(cache["a"]).tolist()
            cache_h["point"]+=np.array(cache["point"]).tolist()
            ujson.dump(cache_h,file)
            cache={"a":[],"point":[],"test":[]}
        cache={"a":[],"point":[],"test":[]}
        