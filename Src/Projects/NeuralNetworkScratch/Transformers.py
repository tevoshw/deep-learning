import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(
    precision=2,    
    sci_mode=False, 
    linewidth=100  
)



data = torch.randint(0, 100, (2, 10))


class TransformerLayer(nn.Module):
    def __init__(self, d_model = 10, qkv = 10, heads = 2):
        super().__init__()
        # QKV Weights
        self.wQ = nn.Linear(d_model, qkv, bias = False)
        self.wK = nn.Linear(d_model, qkv, bias = False)
        self.wV = nn.Linear(d_model, qkv, bias = False)

        # D_k
        self.heads = heads
        self.d_k = d_model // self.heads

        # Bn
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # FFN Weights
        self.w1 = nn.Linear(d_model, d_model * 4, bias = True)
        self.w2 = nn.Linear(d_model * 4, d_model, bias = True)
            # self.b2

    def forward(self, x):
    # BTC
        B, T, C= x.shape
    # Embedding
        x_ori = x
    # QKV
        Q = self.wQ(x)
        K = self.wK(x)
        V = self.wV(x)
    # Heads
        Q = Q.view(B, T, self.heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.heads, self.d_k).transpose(1, 2)
    # Scores
        scores = F.scaled_dot_product_attention(query=Q, key=K, value=V)
        scores = scores.transpose(1, 2).contiguous().view(B, T, -1)
    # Add & Norm 1
        x = self.ln1(scores + x_ori)
    # FFN
        x = self.w2(F.gelu(self.w1(x)))
    # Add e Norm 2 
        x = self.ln2(x + x_ori)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, vocab_size = 100, d_model = 10, seq_len = 100, blocks = 2):
        super().__init__()
    # Define the number of blocks
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.embeddings_positions = nn.Embedding(seq_len, d_model)
        self.transformers = nn.ModuleList([TransformerLayer() for _ in range(blocks)])

    def forward(self, x):
    # The loop of the x
        B, T = x.shape
    # Embedding
        positions = torch.arange(0, T)
        x = self.embeddings(x) + self.embeddings_positions(positions)

        for layer in self.transformers:
            x = layer(x)
        return x
        

model = TransformerBlock()
output = model(data)

print(f'Data before the transformers:\n{data}\n')
print(f'Data after the transformers:\n{output}\n')
print(f'Parameters from the model: {model.state_dict()}')