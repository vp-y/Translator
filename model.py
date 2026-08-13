import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab-size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) *math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int , dropout: float ) -> None :

        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        # create a matrix of shape (seq_len, d_model) to hold the positional encodings

        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len, 1) to hold the position indices
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        #apply the sine and cosine functions to the position indices and the div_term
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = (x + self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # multiplicative parameter alpha initialized to 1
        self.bias = nn.Parameter(torch.zeros(1))  # additive parameter beta initialized to 0

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)   # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)   # W2 and B2

    def forward(self, x):
        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_ff) -> (Batch, Seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"    # d_k = d_model // h

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv
        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k ,v ,mask):
        query = self.w_q(q)  # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)
        key = self.w_k(k)    # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model) 
        value = self.w_v(v)  # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)

        query = query.view(query.shape[0] , query.shape[1] ,self.h , self.d_k).transpose(1, 2)  # (Batch, h, Seq_len, d_k)
        key = key.view(key.shape[0] , key.shape[1] ,self.h , self.d_k).transpose(1, 2)    # (Batch, h, Seq_len, d_k)
        value = value.view(value.shape[0] , value.shape[1] ,self.h , self.d_k).transpose(1, 2)  # (Batch, h, Seq_len, d_k)