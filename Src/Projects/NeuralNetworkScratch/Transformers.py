import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLayer(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, qkv, heads):

        # Embeddings
        self.embeddings = nn.Embeddings(vocab_size, d_model)
        self.embeddings_positions = nn.Embedding(seq_len, seq_len)

        # QKV Weights
        self.wQ = nn.Linear(d_model, qkv)
        self.wK = nn.Linear(d_model, qkv)
        self.wV = nn.Linear(d_model, qkv)

        # D_k
        self.d_k = heads // d_model

        # Bn
        self.bn1 = nn.BatchNorm2d()
        self.bn2 = nn.BatchNorm2()

        # FFN Weights
        self.w1 = nn.Linear(d_model, d_model * 4)
            # self.b1
        self.w2 = nn.Linear(d_model * 4, d_model)
            # self.b2

    def forward(self, x):

    # BTC
        B, T, C = x.shape
        x_ori = x

    # Embedding
        positions = torch.arange(0, T)
        x = self.embeddings(x) + self.embeddings_positions(positions)

    # QKV
        Q = self.wQ(x)
        K = self.wK(x)
        V = self.wV(x)

    # Heads
        Q = Q.view(B, T, self.heads, self.d_k).tranpose(1, 2)
        K = K.view(B, T, self.heads, self.d_k).tranpose(1, 2)
        V = V.view(B, T, self.heads, self.d_k).tranpose(1, 2)

    # Scores
        scores = F.scaled_dot_product_attention(query=Q, key=K, value=V)
        scores = F.softmax(scores, dim = -1)

    # Add & Norm 1
        x = scores + x_ori
        x = self.bn1(x)

    # FFN
        x = self.w1(x)
            #  x = 
        x = self.w2(x)
            # x = 

    # Add e Norm 2 
        x = x + x_ori
        x = self.bn2(x)


class TransformerBlock(nn.Module):
    def __init__(self, blocks):
        
    # Number of blocks
        self.blocks = blocks

    # Define the number of blocks
        self.transformers = nn.ModuleList([TransformerLayer(blocks) for _ in range(blocks)])

    def forward(self, x):

    # The loop of the x
        for layer in self.transformers:
            x = layer(x)
            return x
        

model = TransformerBlock(heads = 8)