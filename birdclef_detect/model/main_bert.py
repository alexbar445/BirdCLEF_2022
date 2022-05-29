import numpy as np
import torch as tc
import os,math,glob,tqdm,ujson,torchaudio,random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from nn_class import my_dic,Adjest_data,load
from torch.nn.modules.transformer import TransformerEncoderLayer,LayerNorm,TransformerEncoder,Optional,Tensor,xavier_uniform_

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "leaky_relu":
        return F.leaky_relu

    raise RuntimeError("activation should be relu/gelu/leaky_relu, not {}".format(activation))

tc.nn.modules.transformer._get_activation_fn=_get_activation_fn


class PositionalEncoding(nn.Module):
    """def __init__(self, d_model, dropout=0.1, max_len=5000):  # ninp, dropout
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len=max_len
        
        pe = tc.zeros(max_len, d_model)  # 5000 * 200
        position = tc.arange(0, max_len, dtype=tc.float).unsqueeze(1)  # [[0],[1],...[4999]] 5000 * 1
        div_term = tc.exp(tc.arange(0, d_model, 2).float() * (-math.log(
            10000.0) / d_model))  # e ^([0, 2,...,198] * -ln(10000)(-9.210340371976184) / 200) [1,0.912,...,(1.0965e-04)]
        pe[:, 0::2] = tc.sin(position * div_term)
        pe[:, 1::2] = tc.cos(position * div_term)
        pe = pe.unsqueeze(0) # 1*5000  * 200, 最长5000的序列, 每个词由1 * 200的矩阵代表着不同的时间
        self.register_buffer('pe', pe)"""
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.max_len=max_len
        self.d_model=d_model
        self.pe = nn.Parameter(tc.zeros(1,max_len,d_model))
        
    def adject(self,max_len):
        self.max_len=max_len
        cache =tc.zeros(1,max_len,self.pe.shape[2],device=self.pe.device)
        cache[:,:self.pe.shape[1]]=self.pe.detach()
        self.pe=nn.Parameter(cache)
    
    def forward(self, x:tc.Tensor):
        if x.shape[1]>self.max_len:
            self.adject(max_len=x.shape[1])
        
        x = x + self.pe[:,:x.shape[1]]        # tc.Size([batch,n,d_model])
        return x

class Bert(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu",
                 layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 device=None, dtype=None) -> None:
        super().__init__()


        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps,)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)



        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                 src_key_padding_mask: Optional[Tensor] = None,) -> Tensor:
        

        if src.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return memory
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                
                xavier_uniform_(p)

class my_bert(nn.Module):
    def __init__(self,bert:Bert,ip_dictionary_length,op_dictionary_length,d_model=512,max_len=2000):
        super().__init__()
        self.save_x=0
        self.pe=PositionalEncoding(d_model,0,max_len)
        self.bert=bert
        self.l1=nn.Linear(ip_dictionary_length,d_model)
        self.sigmoid_layer1=nn.Linear(d_model,op_dictionary_length)
        self.softmax_layer1=nn.Linear(d_model,op_dictionary_length)
        self.sigmoid_layer2=nn.Linear(op_dictionary_length,op_dictionary_length)
        self.softmax_layer2=nn.Linear(op_dictionary_length,op_dictionary_length)

        self.add_module(self.pe._get_name(),self.pe)
        self.add_module(self.bert._get_name(),self.bert)
        
    def forward(self,src: tc.Tensor):
        #print(src.shape,tgt.shape)
        src=self.l1(src)
        src=self.pe.forward(src)
        x=self.bert.forward(src)
        sigmoid_O=self.sigmoid_layer1.forward(x)
        sigmoid_O=F.leaky_relu(sigmoid_O)
        sigmoid_O=self.sigmoid_layer2.forward(sigmoid_O)
        sigmoid_O=tc.sigmoid(sigmoid_O.mean(1))
        
        #softmax_O=self.softmax_layer1.forward(x)
        #softmax_O=F.leaky_relu(softmax_O)
        #softmax_O=self.softmax_layer2.forward(softmax_O)
        #softmax_O=F.softmax(softmax_O.mean(1),dim=-1)
        
        #print(output.shape,"yoyo")
        return sigmoid_O#,softmax_O
    
    #def load(self,path):
    #    cache=tc.load(path)
    #    self.load_state_dict(cache["state_dict"])
     
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
                    print(k,v.shape)
                    new_dict[k]=v
                else :
                    
                    try:
                        print(f"pass: {k} data_v: {v.shape} model_v: {model_dict[k].shape}")
                    except:
                        
                        print(f"pass: {k} data_v: {v.shape} model_v: None")
                    optimizer=tc.optim.Adam(self.parameters(),lr=1e-4)
                    
                    s=v.shape
                    if len(model_dict[k].shape)==4:
                        model_dict[k][:s[0],:s[1],:s[2],:s[3]]=v
                        new_dict[k]=model_dict[k]
                    elif len(model_dict[k].shape)==3:
                        model_dict[k][:s[0],:s[1],:s[2]]=v
                        new_dict[k]=model_dict[k]
                    elif len(model_dict[k].shape)==2:
                        model_dict[k][:s[0],:s[1]]=v
                        new_dict[k]=model_dict[k]
                    elif len(model_dict[k].shape)==1:
                        model_dict[k][:s[0]]=v
                        new_dict[k]=model_dict[k]
                    
                    print(v.shape)
                    print(model_dict[k].shape) 
                    print("save_data_to_new")
                    input("enter?")
                    
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
        #return model#, optimizer


    def save(self,optimizer:tc.optim.Adam,path=None,loss=None):
        if path:
            tc.save({"state_dict":self.state_dict()},path)
        else:
            if "model_data" not in glob.glob("*"):
                os.system("mkdir model_data")
            if loss==None:
                tc.save({"data":self.save_x,"state_dict":self.state_dict(),"optimizer":optimizer.state_dict()
                         },"model_data/"+str(self.save_x)+".pth")
            else :
                tc.save({"data":self.save_x,"state_dict":self.state_dict(),"optimizer":optimizer.state_dict()
                         },"model_data/"+str(self.save_x)+" %2f.pth"%loss)
                
            self.save_x+=1

########################################################################################################################################################################
if __name__ =="__main__":
    dc=my_dic()
    try: 
        data_path="./"
        adjest_data=Adjest_data(dc,data_path+"filer_data/finish_data/noise.wav")
        data={"batch":5,"skip":2,"device":"cuda:0","classnum":21}
    except:
        data_path="D:/python_data/voice/birdclef_voice/"
        adjest_data=Adjest_data(dc,data_path+"filer_data/finish_data/noise.wav")
        data={"batch":5,"skip":2,"device":"cuda:0","classnum":21}
        
    bert=Bert(512,activation="leaky_relu",num_encoder_layers=6,dim_feedforward=2048,dropout=0.02)
    bert=my_bert(bert,257,21,512)
    bert=bert.to(data["device"])

    
    try:
        adjest_data.load(data_path+"filer_data/finish_data")
    except:
        x_trains=glob.glob(data_path+"filer_data/x_train*.pt")
        y_trains=glob.glob(data_path+"filer_data/y_train*.json")
        for y_path in y_trains:
            with open(y_path,"r") as file:
                y_train=ujson.load(file)["y"]
            for x in y_train:
                for t in x:
                    dc.ck(t) 
            dc.data.sort()
        for x in range(len(x_trains)):
            x_train,y_train=load(x_trains[x],y_trains[x],dc)
            adjest_data.add_data(x_train,y_train)
            print("loaded ",x_trains[x])
        
        adjest_data.cut()
        adjest_data.Data_Augmentation()
        adjest_data.save(data_path+"filer_data/finish_data")


    #optimizer=tc.optim.SGD(bert.parameters(),lr=1e-5,momentum=0.9)
    optimizer=tc.optim.Adam(bert.parameters(),lr=1e-6,betas=(0.9,0.9))#,weight_decay=0.0001)
    try:
        bert.load_checkpoint("model_data/main.pth",optimizer=optimizer)
        print("load model fanish")
    except:
        print("error")

    print(optimizer.param_groups[0]['lr'])
    optimizer.param_groups[0]['lr'] = 5e-7
    print(optimizer.param_groups[0]['lr'])

    a,point=0,0

    loss_fc=nn.BCELoss(reduction="sum")
    
    #maindata
    X_test,Y_test=adjest_data.load_data(1000,"test",0.001)
    cache={"a":[],"point":[],"test":[]}
    for epoch in range(2000):
        X_train,Y_train=adjest_data.load_data(4000,"train",0.001)
        
        data["len"]=X_train.shape[0]
        bert=bert.train()
            
        print("\n")
        for i in range(int(len(X_train)/data["batch"])): 
            x_train=X_train[i:i+data["batch"]].to(data["device"])
            y_train=Y_train[i:i+data["batch"]].to(data["device"])
            
            optimizer.zero_grad()
            sigmoid_O,softmax_O=bert.forward(x_train)
            
            for x in range(data["batch"]):
                cache1=(sigmoid_O[x]>0.5).cpu()
                cache1=(cache1.cpu()[y_train[x].to(tc.bool)]).to(tc.float32)
                if cache1.numel()!=0:
                    point+=cache1.sum()/cache1.numel()/data["batch"]
                else:
                    point+=1/data["batch"]
                    

            loss=loss_fc.forward(sigmoid_O,y_train)/data["batch"]
            loss.backward()
            optimizer.step()

            a+=loss.sum()
            
            if i==0:
                time=tqdm.tqdm(total=data["len"],desc="hi")
                
                a,point=0,0
            elif i%data["skip"]==0:
                
                cache["a"]+=[a.cpu().tolist()/data["skip"]]
                cache["point"]+=[point.tolist()/data["skip"]]
                time.set_description("loss: %.3f , point: %.3f"%(np.array(cache['a']).mean(),
                                                            np.array(cache['point']).mean()))
                time.update(data["skip"]*data["batch"])
                a,point=0,0
        bert.save(optimizer,loss=np.array(cache["a"][-int(len(X_train)/data["batch"])-11:]).mean())
        
            
        del x_train,y_train,X_train,Y_train,sigmoid_O,softmax_O
        
        bert=bert.train(False)

        time=tqdm.tqdm(total=X_test.shape[0],desc="hi")
        loss=[]
        print("\n")
        for i in range(int(len(X_test)/data["batch"])): 
            x_test=X_test[i:i+data["batch"]].to(data["device"])
            y_test=Y_test[i:i+data["batch"]].to(data["device"])
            sigmoid_O,softmax_O=bert.forward(x_test)
            for x in range(data["batch"]):
                cache1=(sigmoid_O[x]>0.5).cpu()
                cache1=(cache1.cpu()[y_test[x].to(tc.bool)]).to(tc.float32)
                if cache1.numel()!=0:
                    cache["test"].append(cache1.sum()/cache1.numel()/data["batch"])
                else:
                    cache["test"].append(1/data["batch"])
            
            loss.append(loss_fc.forward(sigmoid_O,y_test).cpu().tolist()/data["batch"])

            if i%data["skip"]==0:
                time.set_description("test->loss: %.3f , point: %.3f"%(np.array(loss).mean(),
                                                            np.array(cache["test"]).mean()))
                time.update(data["skip"]*data["batch"])
                
        del x_test,y_test
        """try :
            with open("data.json" ,"r") as file:
                cache_h=ujson.load(file)
        except:
            cache_h={"a":[],"point":[],"test":[]}

        with open("data.json" ,"w") as file:
            cache_h["test"]+=cache["test"]
            cache_h["a"]+=cache["a"]
            cache_h["point"]+=cache["point"]
            ujson.dump(cache_h,file)
            cache={"a":[],"point":[],"test":[]}"""
    
