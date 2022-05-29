import numpy as np
import torch as tc
import os,math,glob,tqdm,ujson,librosa,random,time
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

def preprocess_voice(voice_path,torch=False):
    if torch:
        audio=tc.load(voice_path).numpy()
    else:
        audio, _ = librosa.core.load(voice_path, sr=None, mono=True)

    melspec = librosa.feature.melspectrogram(
        audio,
        sr=32_000,
        n_fft=800,
        hop_length=320,
        n_mels=128,
        fmin=20,
        fmax=14_000,
        power=1,
    )
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
    return tc.from_numpy(x_train.T)

class testDataset(tc.utils.data.Dataset):
    """Some Information about MyDataset"""
    def __init__(self,paths:list,filepath:str,fit_randn=0.01):
        super().__init__()
        if not isinstance(paths,np.ndarray):
            paths=np.array(paths,dtype=np.object0)
        self.filepath=filepath
        self.paths=paths
        self.randn=fit_randn

    def preprocess_voice(self,voice_ndarray):
        melspec = librosa.feature.melspectrogram(
            voice_ndarray,
            sr=32_000,
            n_fft=800,
            hop_length=320,
            n_mels=128,
            fmin=20,
            fmax=14_000,
            power=1,
        )
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
        return tc.from_numpy(x_train.T)
    
    def __getitem__(self, index):
        def get(index):
            return self.paths[index]
        
        path=get(index)
        path=self.filepath+path+".pt"
        if os.path.isfile(path):
            voice_data=tc.load(path).numpy()
            voice_data=self.preprocess_voice(voice_data)
            
        else: 
            return tc.randn(500,124,dtype=tc.float32)*0.01
        
        """more Data_Augmentation"""
        #voice_data.shape=(500,128)
        a=1
        cache=np.random.randint(a,4-a)
        voice_data=voice_data[:,cache:-4+cache]#voice_data.shape=(500,120)#-(8-cache)

        
        voice_data=(voice_data-voice_data.min())/(voice_data.max()-voice_data.min())
        d1=tc.var(voice_data)**0.5
        cache=tc.randn(voice_data.shape)*d1*self.randn#0.01
        voice_data+=cache#0.01
        voice_data=(voice_data-voice_data.min())/(voice_data.max()-voice_data.min())
        if voice_data.shape[0]<500:
            cache=tc.randn(500,124,dtype=tc.float32)*d1*self.randn
            cache[:voice_data.shape[0]]=voice_data.shape[0]
            voice_data=cache
        return voice_data.to(tc.float32)[:500]
    
    def __len__(self):
        return len(self.paths)

def load(sorce_file,targer_file,voice_path,data_set:list,len_=5*32000):
    """voice_path had not add '.ogg'"""
    audio, _ = librosa.core.load(sorce_file+voice_path+".ogg", sr=None, mono=True)
    x=0
    while audio.shape[0]>len_:
        cache=audio[:len_]
        tc.save(tc.from_numpy(cache).to(tc.float16),targer_file+voice_path+f"_{x}"+".pt")#len=2000
        data_set.append(voice_path+f"_{x}")
        audio=audio[len_:]
        x+=1
    tc.save(tc.from_numpy(audio).to(tc.float16),targer_file+voice_path+f"_{x}"+".pt")#len=2000
    data_set.append(voice_path+f"_{x}")

from nn_class import load_dataset,my_dic,count_point,set_scored_data,test_model
from main_ResNet50 import Model,Model_p

if __name__ =="__main__":
    data={"batch":1*4,"skip":2,"device":"cuda:0","train_time":1000,"test_time":1000,"mode":"ho1me","n":500*1,"save_time":2,
            "num_workers":5,"randn_rate":0.1,"mix_rate":0.01,"pretrain_epoch":200,"train_epoch":10,"data_mode":"data",
            "train_time":4000,"test_time":1000,"train_time_m":3,"num_workers":5}
    if data["mode"]=="home":
        #data["device"]="cpu"
        data["model_path"]="model_data/more10p5_p50_fmore3p9_p74_1.pth"
        data["dc_path"]="D:/python_data/voice/birdclef_voice/voice_data/"
        data["train_path"]="D:/python_data/voice/birdclef_voice/voice_data/train_data/"
        data["test_path"]="D:/python_data/voice/birdclef_voice/voice_data/test_data/"
        data["scored_path"]="D:/python_data/voice/birdclef_voice/scored_birds.json"
        data["info_path"]="./"
    else: 
        data["model_path"]="../input/cache-data/more10p5_p50_fmore3p9_p74_1.pth"
        data["dc_path"]="../input/data123/"
        data["train_path"]="../input/train-data/train_data/"
        data["test_path"]="../input/test-data/test_data/"
        data["scored_path"]="../input/birdclef-2022/scored_birds.json"
        data["info_path"]="../working/"
        data["target_filepath"]="../working/test_soundscapes/"
        data["sorce_filepath"]="../input/cache-data/test_soundscapes/"
        if not os.path.exists(data["target_filepath"]):
            os.mkdir(data["target_filepath"])
        data["num_workers"]=2
    #os.system("cp -r ../input/birdclef-2022 ../working")

    data_path=["../input/data123/","../input/data123/more10p5_p50_fmore3p9_p74.pth","../input/cache-data/"]
    #data_path=["../input/data123/","../input/data123/more10p5_p50_fmore3p9_p74.pth","../input/birdclef-2022/"]
    #data_path=["D:/python_data/voice/birdclef_voice/voice_data/","model_data/more10p5_p50_fmore3p9_p74.pth","D:/python_data/voice/birdclef_voice/voice_data/score_data/"]
    dc=my_dic()
    if os.path.exists(data["dc_path"]):
        dc.load(data["dc_path"])
    else :
        with open(data["dc_path"]+"data.json","r") as file:
            y_train=ujson.load(file)["y"]
        for x in y_train:
            for t in x:
                dc.ck(t) 
        dc.data.sort()
        dc.save(data["dc_path"])
    test=pd.read_csv(data_path[2]+"test.csv",encoding="UTF-8")

    row_id=test["row_id"]
    file_id=test["file_id"]
    end_time=test["end_time"]
    hs_voice_path=""
    bird=test["bird"]

    if os.path.exists(data["target_filepath"]+"test.json") and False:
        with open(data["target_filepath"]+"test.json","r") as file:
            test_paths=ujson.load(file)
    else:
        test_paths=[]
        time=tqdm.tqdm(total=len(file_id),desc="hi")
        for voice_path in file_id:
            time.update(1)
            print(voice_path)
            try:
                load(sorce_file=data["sorce_filepath"],targer_file=data["target_filepath"],voice_path=voice_path,data_set=test_paths)
            except:
                pass
        with open(data["target_filepath"]+"test.json","w") as file:
            print(test_paths)
            ujson.dump(test_paths,file)
    
        
    model = Model(Model_p,0.5)
    model.load_checkpoint(data["model_path"])
    model=model.to(data["device"])

    train_dataset,_=load_dataset(data,dc,type=data["data_mode"])
    test_dataset=testDataset(test_paths,data["target_filepath"],0.2)
    test_dataset = tc.utils.data.DataLoader(test_dataset, batch_size=data["batch"],shuffle=True,num_workers=data["num_workers"])#構建tc資料集
    
    cache1=train_dataset.dataset
    cache2=test_dataset.dataset
    #cache2.re_set(data["test_time"],repeat=2,mode=True,add_empty=True,batch=data["batch"])
    
    with open(data["scored_path"],"r")as file:
        data["scored"]=ujson.load(file)
    data["scored"]=dc.transform(data["scored"]).sum(0).to(tc.bool)
    set_scored_data(train_dataset,mask=data["scored"])
    cache1.re_set(data["test_time"],repeat=2,mode=True,add_empty=True,batch=data["batch"])
    
    
    model.optimizer.param_groups[0]["betas"]=(0.9,0.999)
    model.optimizer.param_groups[0]['lr'] = 1e-5
    model.optimizer.param_groups[0]['weight_decay'] = 0.1
    for i in range(2):
        run_info={"train_loss":[],"train_point":[],"test_loss":[],"test_point":[]}
        model.train()
        time=tqdm.tqdm(total=len(cache1)*2,desc="hi")
        dataset1=iter(train_dataset)
        dataset2=iter(test_dataset)
        while True:
            try:
                x1,y1=next(dataset1)
                x2=next(dataset2)
                #print(x1.shape,x2.shape)
                if x1.shape[0]!=data["batch"] and x2.shape[0]!=data["batch"]:
                    break
                x=tc.cat((x1.to(data["device"]),x2.to(data["device"])))
                y=tc.cat((y1.to(data["device"]),tc.zeros(y1.shape,device=data["device"])))
                cache=tc.zeros(y.shape[0],device=data["device"])
                cache[:y1.shape[0]]=1
                y=tc.cat((y,cache[:,None]),dim=1)
            except StopIteration:
                break
            output=model.train_all(x,y,mask=data["scored"])
            loss=output["loss"]
            output=output["output"]
            run_info["test_loss"].append(loss.cpu().tolist())
            run_info["test_point"].append([count_point(output[:y1.shape[0]],y[:y1.shape[0],:-1])])
            
            time.set_description(f"loss(d,c): {np.array(run_info['test_loss']).mean(axis=0)} point(train,test): {np.array(run_info['test_point']).mean(axis=0)}")
            time.update(data["batch"]*2)
        model.optimizer.zero_grad(True)
        #test_model(model=model,data=data,dataset=test_dataset,run_info=None)#updata run_info,model

    def run(x,index):
        if x.shape[-1]==128:
            x=x[...,2:-2]
        with tc.no_grad():
            sigmoid_O=model.cnn.forward(x.to(data["device"]))[:,data["scored"]]
            sigmoid_O=sigmoid_O>0.017

        return sigmoid_O[index.to(tc.bool)].cpu()

    voice_error=0
    #in test error
    import time
    start=time.time()
    target=[]
    row_id=test["row_id"]
    file_id=test["file_id"]
    end_time=test["end_time"]
    hs_voice_path=""
    voice_set=[]
    bird=test["bird"]
    bird_index=[]
    data["batch"]=300

    for x in range(len(file_id)):
        time_=int(end_time[x])#s
        path=data["target_filepath"]+file_id[x]+f"_{(time_-5)//5}"+".pt"
        if os.path.exists(path):
            voice=preprocess_voice(path,torch=True).to(tc.float32)
            voice=(voice-voice.min())/(voice.max()-voice.min())
            d1=tc.var(voice)**0.5
            cache=tc.randn(500,124,dtype=tc.float32)*d1*data["randn_rate"]
            if voice.shape[0]<500:
                cache[:voice.shape[0]]+=voice[:,2:-2]
            else:
                cache+=voice[:500,2:-2]
            voice=cache
            voice=(voice-voice.min())/(voice.max()-voice.min())
        else: 
            print("error")
            voice=tc.randn(500,124,dtype=tc.float32)*0.1
        
        voice_set.append(voice)
        bird_index.append(bird[x])

        if x%data["batch"]==0 and len(voice_set)>1:
            voice_set=tc.stack(voice_set)

            bird_index=dc.transform(bird_index)
            a=run(voice_set,bird_index)
            voice_set=[]
            bird_index=[]
            target+=a.tolist()
    if len(voice_set)>=1:
        if len(voice_set)>1:
            voice_set=tc.stack(voice_set)
        else: 
            voice_set=voice_set[0][None]

        bird_index=dc.transform(bird_index)
        a=run(voice_set,bird_index)
        target+=a.tolist()

    end = time.time()
    print(end-start)

    output=pd.DataFrame({"row_id":row_id,"target":target})
    output=output.set_index('row_id')
    output.to_csv("submission.csv")
    print("fanish")
    print(output[:3])
    