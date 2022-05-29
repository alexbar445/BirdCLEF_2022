"""main_load"""
import numpy as np
import math,ujson,glob,torchaudio,tqdm,os
import matplotlib.pyplot as plt
import torch as tc
import soundfile as sf
import pandas as pd
import librosa
from nn_class import my_dic,MyDataset
def load_data(cache_path,birdclef_path="D:/python_data/voice/birdclef_voice/",target_path="D:/python_data/voice/birdclef_voice/voice_data/",len_=2000,run_pcen=False):
    
    data=pd.read_csv(birdclef_path+"train_metadata.csv",encoding="UTF-8")
    for x in ["common_name","type","latitude","longitude","scientific_name","author","license","time","url"]:
        del data[x]
    data=data[data["rating"]!=0]
    p=data["primary_label"].to_list()
    pp=data["secondary_labels"].to_list()

    for i in range(len(pp)):
        pp[i]=list(pp[i])
        for x in ["[","]",",","'"]:
            while pp[i].count(x):
                pp[i].remove(x)
        pp[i]="".join(pp[i])
        pp[i]=pp[i].split(" ")+[p[i]]
        if pp[i].count(''):
            pp[i].remove('')
        
    del data["primary_label"],data["secondary_labels"]

    data.index=range(len(data))
    data["lable"]=pp

    tp=data["lable"].copy()
    for y in range(len(data["lable"])):
        tp[y]=len(data["lable"][y])
        if y%100==0: 
            print(y/len(data["lable"]))
    tp=np.array(tp,dtype=np.bool8)
    data_set=data[tp]#去除空[]
    del data

    data_set["old_name"]=data_set["filename"].copy()

    for i in range(len(data_set)):
        #print(data_set["old_name"][i],len(data_set["old_name"]),i)
        if i%100==0: 
            print(i/len(data_set))
        if data_set["old_name"][i].count("/")==0:
            #print(birdclef_path+"train_audio/*/"+data_set["old_name"][i])
            data_set["filename"][i]=glob.glob(birdclef_path+"train_audio/*/"+data_set["old_name"][i])[0]
            continue
        else: data_set["filename"][i]=birdclef_path+"train_audio/"+data_set["filename"][i]
        n=data_set["old_name"][i].index("/")
        #print(data_set["old_name"][i],data_set["old_name"][i][n+1:-4])
        data_set["old_name"][i]=data_set["old_name"][i][n+1:-4]
        
    #data_set["filename"]=birdclef_path+"train_audio/"+data_set["filename"]

    data_set.index=range(len(data_set))

    ###############更改且獲取類別###############^^
    if os.path.exists(cache_path+"voice_long.json"):
        with open(cache_path+"voice_long.json","r") as file:
            voice_long=ujson.load(file)["0"]
    else:
        time=tqdm.tqdm(total=len(data_set),desc="hi")
        voice_long=np.zeros((len(data_set),2))
        print("load long")
        for i in range(len(data_set)):
            voice_path=data_set["filename"][i]
            data,sample_rate=torchaudio.load(voice_path)

            data=data[0]
            voice_long[i]=[data.shape[0]/sample_rate,i]
            if i%100==0 and i!=0:
                time.update(100)
        cache=np.argsort(voice_long[:,0])
        voice_long=voice_long[cache]
        #print(voice_long[:10])
        #print(cache)
        with open(cache_path+"voice_long.json",mode="w") as file:
            ujson.dump({"0":voice_long.tolist()},file)
    
    ###########################獲取長度
    data_train={"x":[],"y":[]}

    y,n=0,0
    time=tqdm.tqdm(total=len(pp),desc="hi")

    #voice_long.reverse()

    for o,i in voice_long:
    #for i in range(len(data_set["filename"])):
        #i=int(i)
        voice_path=data_set["filename"][i]
        audio, _ = librosa.core.load(voice_path, sr=None, mono=True)

        melspec = librosa.feature.melspectrogram(
            y=audio,
            sr=32_000,
            n_fft=800,
            hop_length=320,
            n_mels=128,
            fmin=20,
            fmax=14_000,
            power=1,
        )
        if run_pcen:
            x_train=librosa.pcen(
                melspec * (2 ** 31),
                time_constant=0.06,
                eps=1e-6,
                gain=0.8,
                power=0.25,
                bias=10,
                sr=32_000,
                hop_length=320,
            )
            x_train=x_train.T
        melspec=melspec.T

        if melspec.shape[0]<=20:
            print(melspec.shape,voice_path)
            continue
        x=0
        while melspec.shape[0]>len_:#根據原理,ntaick=1s
            if run_pcen:
                cache=x_train[:len_]
                cache=(cache.min()-cache)/(cache.min()-cache.max())
                tc.save(tc.from_numpy(cache).to(tc.float16),target_path+data_set["old_name"][i]+f"_{x}"+".pt")#len=2000
                x_train=x_train[len_:]
            
            cache1=melspec[:len_]
            cache1=(cache1.min()-cache1)/(cache1.min()-cache1.max())
            tc.save(tc.from_numpy(cache1).to(tc.float16),target_path+"O"+data_set["old_name"][i]+f"_{x}"+".pt")#len=2000
            melspec=melspec[len_:]
            
            data_train["x"].append(data_set["old_name"][i]+f"_{x}")
            data_train["y"].append(data_set["lable"][i])
            x+=1
        if run_pcen:
            x_train=(x_train.min()-x_train)/(x_train.min()-x_train.max())
            tc.save(tc.from_numpy(x_train).to(tc.float16),target_path+data_set["old_name"][i]+f"_{x}"+".pt")
        melspec=(melspec.min()-melspec)/(melspec.min()-melspec.max())
        tc.save(tc.from_numpy(melspec).to(tc.float16),target_path+"O"+data_set["old_name"][i]+f"_{x}"+".pt")
        
        data_train["x"].append(data_set["old_name"][i]+f"_{x}")
        data_train["y"].append(data_set["lable"][i])
        
        if y%100==0 and y!=0:
            time.update(100)
        
        if (y%2000==0 and y!=0) or y==len(data_set)-1 :
            with open(target_path+"y_train"+str(n)+".json","w") as file:
                ujson.dump(data_train,file)
            
            data_train={"x":[],"y":[]}
            n+=1

        y+=1
            
    data=glob.glob(target_path+"y_train*")
    cache={"x":[],"y":[]}

    for x in data:
        with open(x,"r") as file:
            data_train=ujson.load(file)
        cache["x"]+=data_train["x"]
        cache["y"]+=data_train["y"]
        os.remove(x)
    with open(target_path+"data.json","w") as file:
        ujson.dump(cache,file)

def load_dc(cache_path,target_path):
    dc=my_dic()
    with open(target_path+"data.json","r") as file:
        y_train=ujson.load(file)["y"]
    for x in y_train:
        for t in x:
            dc.ck(t)
    dc.data.sort()
    dc.save(cache_path)

def run(cache_path="D:/python_data/voice/birdclef_voice/voice_data/",birdclef_path="D:/python_data/voice/birdclef_voice/",target_path="D:/python_data/voice/birdclef_voice/voice_data/",len_=2000):
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    load_data(cache_path=cache_path,birdclef_path=birdclef_path,target_path=target_path,len_=len_)
    load_dc(cache_path=cache_path,target_path=target_path)

def cut(cache_path,sources,target_path,):
    with open(sources+"data.json","r") as file:
        cache=ujson.load(file)
    dc=my_dic()
    dc.load(cache_path)
    cache=MyDataset(cache["x"],cache["y"],dc,sources,reset=True)
    #print(cache.labels.sum(dim=0))
    #input()
    cache.cut(sources_path=sources,rate=0.2,
                train_data_path=target_path+"train_data/",
                test_data_path=target_path+"test_data/")


if __name__ =="__main__":
    cache_path="D:/python_data/voice/birdclef_voice/voice_data/bird_voice/"
    target_path="D:/python_data/voice/birdclef_voice/voice_data/bird_voice/cache/"
    #run(cache_path=cache_path,target_path=target_path,len_=2000)
    cut(cache_path=cache_path,sources=target_path,target_path="D:/python_data/voice/birdclef_voice/voice_data/bird_voice/")