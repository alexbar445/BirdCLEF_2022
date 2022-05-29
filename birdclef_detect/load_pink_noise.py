import numpy as np
import torch as tc
import os,math,glob,tqdm,ujson,warnings,librosa
import torch.nn as nn
import matplotlib.pyplot as plt

def act(x,sr):
    melspec = librosa.feature.melspectrogram(
        x,
        sr=sr,
        n_fft=800,
        hop_length=320,
        n_mels=128,
        fmin=20,
        fmax=14_000,
        power=1,
    )
    """x_train=librosa.pcen(
        melspec * (2 ** 31),
        time_constant=0.06,
        eps=1e-6,
        gain=0.8,
        power=0.25,
        bias=10,
        sr=sr,
        hop_length=320,
    )"""
    return tc.from_numpy(melspec.T)

def load(sorce_file,targer_file,voice_path,data_set:list,len_=5*100,final=".ogg"):
    """voice_path had not add '.ogg'"""
    audio, sr = librosa.core.load(sorce_file+voice_path+final, sr=None, mono=True)
    audio=act(audio,sr)
    x=0
    while audio.shape[0]>len_:
        cache=audio[:len_]
        cache=(cache.min()-cache)/(cache.min()-cache.max())
        tc.save(cache.to(tc.float16),targer_file+voice_path+f"_{x}"+".pt")#len=2000
        data_set.append(voice_path+f"_{x}")
        audio=audio[len_:]
        x+=1
    cache=(cache.min()-cache)/(cache.min()-cache.max())
    tc.save(cache.to(tc.float16),targer_file+voice_path+f"_{x}"+".pt")#len=2000
    data_set.append(voice_path+f"_{x}")


paths=glob.glob("D:/python_data/voice/birdclef_voice/voice_data/pink_noise/*")
time=tqdm.tqdm(total=len(paths),desc="hi")
a="D:/python_data/voice/birdclef_voice/voice_data/pink_noise/"
test_paths=[]
for path in paths:
    time.update(1)
    path=path[len(a):-4]
    print(path)
    load(sorce_file=a,targer_file="D:/python_data/voice/birdclef_voice/voice_data/pink_noise_pt/",voice_path=path,data_set=test_paths,final=".mp3")

with open("D:/python_data/voice/birdclef_voice/voice_data/pink_noise_pt/"+"test.json","w") as file:
    print(test_paths)
    ujson.dump(test_paths,file)
        
    