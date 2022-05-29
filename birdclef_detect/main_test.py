import numpy as np
import torch as tc
import os,math,glob,tqdm,ujson,librosa,random,time
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from nn_class import my_dic
def preprocess_voice(voice_path):
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
########################################################
def run(data_path=["/input/submit/","/input/submit/model/sAs_whole_b4_loss1p5_i124_trandp4_8p4_p34_1p8_p88.pth","/input/birdclef-2022/"],batch=300,trand_rate=0.2,submit_path="submission.csv"
        ):
    tc.backends.cudnn.benchmark = True###new

    data={"batch":batch,"skip":2,"device":"cuda:0","num_workers":2}
    data["device"] = "cuda:0" if tc.cuda.is_available() else "cpu"

    #data_path=["../input/data123/","../input/data123/sAs_whole_b4_loss1p5_i124_trandp4_8p4_p34_1p8_p88.pth","../input/birdclef-2022/"]
    #data_path=["D:/python_data/voice/birdclef_voice/voice_data/","model_data/sAs_whole_b4_loss1p9_i124_trandp65_8p1_p43_1p8_p88.pth","D:/python_data/voice/birdclef_voice/"]


    dc=my_dic() 
    dc.load(data_path[0])
    #print(dc.data)
    with open(data_path[2]+"scored_birds.json","r")as file:
        target=ujson.load(file)
    target=dc.transform(target).sum(0).to(tc.bool)
    dc.data=np.array(dc.data)[target]
    
    data["scored"]=target

    #model = Model()
    #model.load_checkpoint(data_path[1])
    model=tc.load(data_path[1])
    model=model.to(data["device"]).train(False)

    data_path=data_path[2]


    def run(x,index):
        if x.shape[-1]==128:
            x=x[...,2:-2]
        with tc.no_grad():
            sigmoid_O=model.cnn.forward(x.to(data["device"]))[:,data["scored"]]
            #sigmoid_O=tc.softmax(output,dim=1)>0.1
            #sigmoid_O=sigmoid_O>0.135

        return sigmoid_O.cpu()#[index.to(tc.bool)]
    ##############################################################################################################################
    """with tc.no_grad():
        a=model.cnn.forward(tc.randn((data["batch"],500,128),device=data["device"]))
        del a"""

    ##############################################################################################################################

    test=pd.read_csv(data_path+"test.csv",encoding="UTF-8")

    voice_error=0

    #in test error
    start=time.time()
    target=[]
    row_id=test["row_id"]
    file_id=test["file_id"]
    end_time=test["end_time"]
    hs_voice_path=""
    voice_set=[]
    bird=test["bird"]
    bird_index=[]
    for x in range(len(file_id)):
        if hs_voice_path!=file_id[x]:
            print(file_id[x])
            if os.path.isfile(data_path+"test_soundscapes/"+file_id[x]+".ogg"):
                voices=preprocess_voice(data_path+"test_soundscapes/"+file_id[x]+".ogg").to(tc.float32)
                voice_error=0
            else:
                voice_error=1

            hs_voice_path=file_id[x]

        if voice_error:
            #print("voice_error")
            voice=tc.randn(500,128,dtype=tc.float32)*0.1
        else:
            time_=int(end_time[x])#s
            if voices.shape[0]<time_*100:
                #print("add error")
                voice=voices[-500:]
            else :
                voice=voices[time_*100-500:time_*100]

        voice=(voice-voice.min())/(voice.max()-voice.min())
        d1=tc.var(voice)
        voice=voice+tc.randn(voice.shape)*d1**0.5*trand_rate#0.01
        voice=(voice-voice.min())/(voice.max()-voice.min())
        voice_set.append(voice)

        bird_index.append(bird[x])

        if x%data["batch"]==0 and len(voice_set)>1:
            voice_set=tc.stack(voice_set)

            #bird_index=dc.transform(bird_index)
            a=run(voice_set,bird_index)
            voice_set=[]
            #bird_index=[]
            if isinstance(target,list):
                target=a
            else:
                target=tc.cat((target,a),dim=0)
    if len(voice_set)>=1:
        if len(voice_set)>1:
            voice_set=tc.stack(voice_set)
        else: 
            voice_set=voice_set[0][None]

        #bird_index=dc.transform(bird_index)
        a=run(voice_set,bird_index)
        if isinstance(target,list):
            target=a
        else:
            target=tc.cat((target,a),dim=0)
        
    end = time.time()
    print(end-start)
    
    plt.imshow(target[:10])
    plt.show()
    
    percentage=0.35#x in target#0.02 i think is good number :1/21/2......

    """with open("../input/data123/"+"score_bird_rate.json","r") as file:
        t=ujson.load(file).values()
        t=list(t)"""
    cache=target.sort(dim=0)
    
    
    for x in range(target.shape[1]):
        #n=math.ceil(t[x]*target.shape[0])
        n=math.ceil(percentage*target.shape[0])
        target[:,x]=False
        target[cache[1][-n:,x],x]=True
    
    plt.imshow(cache[0][:10])
    plt.show()
    plt.imshow(target[:10])
    plt.show()
    bird_index=dc.transform(bird_index)
    target=target[bird_index.to(tc.bool)].to(tc.bool)
    
    output=pd.DataFrame({"row_id":row_id,"target":target})
    output=output.set_index('row_id')
    output.to_csv(submit_path)
    print("fanish")
    print(output[:3])
    if data["batch"]!=300:
        print("batch error",data["batch"])
if __name__=="__main__":
    run()