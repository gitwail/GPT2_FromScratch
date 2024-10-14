from dataclasses import dataclass
import torch
import torch.nn as nn 
from torch.nn import functional as F
import math
import tiktoken



class MLP(nn.Module):
    def __init__(self,config):
       super().__init__()
       self.c_fc=nn.Linear(config.n_embd,config.n_embd*4)
       self.gelu=nn.GELU(approximate='tanh')
       self.c_proj=nn.Linear(config.n_embd*4,config.n_embd)
       
    def forward(self,x):

        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        self.config=config
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))

    def forward(self,x):
        # x shape is B,T,C
        B,T,C=x.shape
        qkv=self.c_attn(x)
        q,k,v=torch.split(qkv,self.config.n_embd,dim=-1)
        q=q.view(B,T,self.config.n_head,C//self.config.n_head).transpose(1,2)# B,nh,T,hs
        k=k.view(B,T,self.config.n_head,C//self.config.n_head).transpose(1,2)# B,nh,T,hs
        v=v.view(B,T,self.config.n_head,C//self.config.n_head).transpose(1,2)# B,nh,T,hs
        #B,nh,T,hs @B,nh,hs,T -->B,nh,T,T
        att=q@k.transpose(-1,-2)*((k.size(-1))**-0.5)
        att=att.masked_fill(self.bias[:,:,T,T]==0,float('inf'))
        att=F.softmax(att,dim=-1)
        # B,nh,T,T @ B,nh,T,hs --> B,nh,T,hs
        y=att@v
        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.c_proj(y)
        return y



class Block(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.ln_1=nn.LayerNorm(config.n_embd)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.mlp=MLP(config)
        self.config=config

    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x 





class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config

        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.block_size,config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        
        ))
        self.lm_head=nn.Linear(self.config.n_embd,self.config.vocab_size,bias=False)

    def forward(self,idx,targets=None):
        # idx(B,T)
        B,T=idx.size()
        tok_embd=self.transformer.wte(idx) # B,T,C
        pos_embd=self.transformer.wpe(torch.arange(T,device=idx.device)) #B,T,C (B)
        x=tok_embd+pos_embd
        for block in self.transformer.h:
            x=block(x)

        x=self.transformer.ln_f(x)
        logits=self.lm_head(x) # B,T,V
        B,T,V=logits.shape
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(B*T,V),targets.view(B*T))
        return logits,loss
    

    def sample(self,idx,sample_size):
        while idx.size(1)<self.config.block_size and idx.size(1)<sample_size: 
            x= self(idx) # B,T,V
            proba=F.softmax(x[:,-1,:],dim=-1) # B,V
            topk_proba,topk_indices=torch.topk(proba,50,-1)
            ix = torch.multinomial(topk_proba, 1)
            xcol=torch.gather(topk_indices,-1,ix)
            idx=torch.cat((idx,xcol),dim=-1)


        return idx


@dataclass
class GPTConfig:
    block_size: int =1024
    vocab_size: int= 50257
    n_layer:int =12
    n_head:int=12
    n_embd: int =768




###### device
device="cpu"
if torch.cuda.is_available():
    device="cuda"
print(f"the device used: {device}")


#######tokenisation

bpe_tokenizer = tiktoken.get_encoding("gpt2")
# Encode a string into tokens (token ids)
def encode_text(text):
    return bpe_tokenizer.encode(text)

# Decode tokens back into a string
def decode_tokens(tokens):
    return bpe_tokenizer.decode(tokens)



########## model and inference 

# config
config=GPTConfig()
print(config)

# model
model = GPT(config=config).to(device)  # Place the model on the device
model.eval()

################ training the model

with open("input.txt") as f:
    text=f.read()

B,T=4,32
encoded_text=torch.tensor(encode_text(text),dtype=torch.long).to(device)

buf=encoded_text[:B*T+1]
x=buf[:-1].view(4,32)
y=buf[1:].view(4,32)

logits,loss=model(x,y)
print(loss)





################ generation for a sentence

# return_sequence=5
# text="hello everyone my name is gpt2"
# x = torch.tensor(encode_text(text), dtype=torch.long).to(device)
# x=x.unsqueeze(0).repeat(return_sequence,1)
# print(x.shape)

# gen_tokens=model.sample(x,sample_size=15)

# for i in range(5):
#     print(decode_tokens(gen_tokens[i,:].tolist()))
