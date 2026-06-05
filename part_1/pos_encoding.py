import math
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__() #Run the setup code inside nn.Module first
        self.emb = nn.Embedding(max_len, d_model) #Create a learnable lookup table that stores one vector for each position

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)#(device=x.device=Suppose x is on the GPU,Then pos must also be on the GPU)
        pos_emb = self.emb(pos)  
        return x + pos_emb.unsqueeze(0)#(Word information + Position information = Word that knows its position)  

class SinusoidalPositionalEncoding(nn.Module):
    """
    This class creates a big table of sin and cos wave patterns, 
    gives each position a unique vector, 
    and adds that vector to each word so the Transformer knows the order of the words.
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        return x + self.pe[:T].unsqueeze(0)#(Word vector+Position vector=Word knows where it is)