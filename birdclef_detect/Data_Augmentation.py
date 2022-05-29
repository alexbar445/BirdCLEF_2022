"""Data_Augmentation"""
import numpy as np
import torch as tc
import os,math,glob,tqdm,ujson,librosa,random
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from nn_class import my_dic,MyDataset


#path="D:/python_data/voice/birdclef_voice/voice_data/"
def data_augmentation(path:str,least=300):
    with open(path+"x_data.json","r") as file:
        paths=ujson.load(file)
        paths=np.array(paths,dtype=np.object0)
    
    labels=tc.load(path+"y_data.pt").to(tc.bool)
    #with open(path+"data.json","r") as file:
    #    cache=ujson.load(file)
        
    dc=my_dic()
    dc.load("D:/python_data/voice/birdclef_voice/voice_data/")
    #dataset=MyDataset(cache["x"],cache["y"],dc,path,reset=True)
    #d=tc.stack((,tc.arange(dataset.labels.shape[1])))
    d,index=labels.sum(0).sort()

    y_data=labels
    x_data=paths
    #print(y_data[:,130].sum())
    def get(n=1,max=300):
        cache={}
        for x in range(d.shape[0]):
            #print(index[x],"yo")
            if d[x]<=max:
                #print(y_data.shape)
                index_=tc.all(tc.stack((y_data[:,index[x]],y_data.sum(1)<=n)),dim=0)
                #print(y_data[:,index[x]].sum())
                if index_.sum()<10:
                    index_=y_data[:,index[x]]
                cache[index[x].tolist()]=x_data[index_]
        return cache



    count=get(2)
    k,v_path=list(count.keys()),list(count.values())

    #################################################################################get data_path and answer
    data=[]
    #answer=[]
    for x in range(len(v_path)):
        for y in range(len(v_path[x])):
            if v_path[x][y] not in data:
                data.append(v_path[x][y])
                #a=tc.zeros(y_data.shape[1])
                #a[k[x]]=1
                #answer.append(a)
    #answer=tc.stack(answer).to(tc.bool)
    data=np.array(data)
    #################################################################################load data from path
    print(data.shape)

    voice=[]
    answer=[]
    for x in data:
        print("load",f"O{x}")
        index=(paths==x).argmax()
        pp=tc.load(f"D:/python_data/voice/birdclef_voice/voice_data/data/O{x}.pt").to(tc.float32)
        voice.append(pp)
        answer.append(labels[index])
    
    answer=tc.stack(answer).to(tc.bool)
    voice=np.array(voice)
    print(voice.shape)
    print(answer.shape)
    ################################################################################# Data_Augmentation


    h=[]
    sum=answer.sum(0)
    save_data={'x':[],"y":[],"index":0}
    for x in range(answer.shape[1]-1):
        con=True
        while sum[x]<least and sum[x]>1 and con:
            if voice[answer[:,x]].shape[0]==0:
                break
            #h=[]
            i=np.random.randint(voice[answer[:,x]].shape[0])
            data1=voice[answer[:,x]][i]
            a1=answer[answer[:,x]][i]
            #############################################################################################
            #index=np.random.randint(x+1,answer.shape[1])
            index=np.random.randint(0,answer.shape[1])
            if answer[:,index].sum()==0:
                h.append(index)
                if index in h and h.count(index)>3:
                    sum[x]+=1
                continue
            h=[]
            ##############################################################
            i=np.random.randint(voice[answer[:,index]].shape[0])
            data2=voice[answer[:,index]][i]
            a2=answer[answer[:,index]][i]

            
            if abs(data2.shape[0]-data1.shape[0])>200 and data2.shape[0]==data1.shape[0]:
                print("pass")
                continue
            
            
            sum[x]+=1
            sum[index]+=1
            data1=data1.to(tc.float32)
            data2=data2.to(tc.float32)
            if data1.max()!=1 or data1.min()!=0:
                data1=(data1-data1.min())/(data1.max()-data1.min())
            if data2.max()!=1 or data2.min()!=0:
                data2=(data2-data2.min())/(data2.max()-data2.min())
                
            if data1.shape[0]<data2.shape[0]:
                index=np.random.randint(0,data2.shape[0]-data1.shape[0])
                data3=data2.clone()
                data3[index:data1.shape[0]+index]=(data1+data3[index:data1.shape[0]+index])/2
            elif data1.shape[0]>data2.shape[0]:
                index=np.random.randint(0,data1.shape[0]-data2.shape[0])
                data3=data1.clone()
                data3[index:data2.shape[0]+index]+=(data2+data3[index:data2.shape[0]+index])/2
            else :
                data3=(data1+data2)/2
            data3=(data3-data3.min())/(data3.max()-data3.min())
            
            a3=a1+a2
            
            name=f"more{save_data['index']}"
            
            tc.save(data3.to(tc.float16),f"{path}{name}.pt")
            save_data["x"].append(name)
            save_data["y"].append(a3)
            save_data["index"]+=1
                
        print(x/answer.shape[1])
    print("fanish")

    ################################################################################# save data to more.json
    #new_answer=tc.stack(new_answer)

    #name_="more"
    """cache={'x':[],"y":[]}
    for x in range(len(new_data)):
        name=name_+str(x)
        tc.save(new_data[x].to(tc.float16),f"{path}{name}.pt")
        cache['x'].append(name)
        #cache["y"].append(new_answer[x])
        #t=math.ceil(new_data[x].shape[0]/500)
        #cache['x']+=[name]*t
        #cache['y']+=[dc.untransform(new_answer[x]).tolist()]*t
        
        new_data[x]=0 #del
        if x%100==0:
            print(x/len(new_data))"""


    x_data=x_data.tolist()+save_data["x"]
    y_data=tc.cat((y_data,tc.stack(save_data["y"])),dim=0)

    with open(path+"x_more.json","w") as file:
        ujson.dump(x_data,file)
    tc.save(y_data.to(tc.bool),path+"y_more.pt")
    #################################################################################