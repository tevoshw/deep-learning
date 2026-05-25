import torch
import torch.nn.functional as F


torch.set_printoptions(
    precision=2,    
    sci_mode=False, 
    linewidth=100  
)

phrase = torch.rand(5, 20)


"""
SEQ_LEN = 5
D_MODEl = 20
HEADS = 4
BLOCKS = 1


"""


class DecoderMHA:
    def __init__(self, d_model = 20, heads = 4):
        self.Wq = torch.rand(20, 20)
        self.Wk = torch.rand(20, 20)
        self.Wv = torch.rand(20, 20)
        self.Wo = torch.rand(20, 20)


        self.heads = heads
        self.d_k = d_model // heads

        # FFN
        self.W1 = torch.rand(20, 80)
        self.W2 = torch.rand(80, 20)

        self.bias = torch.rand(80)
        self.bias2 = torch.rand(20)

        # Norm
        self.norm1 = torch.nn.LayerNorm(20)
        self.norm2 = torch.nn.LayerNorm(20)


    def forward(self, x):
        T, C = x.shape

        # Get the QKV
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv 
        print(f'Shape of QKF before the split: {Q.shape}, {K.shape}, {V.shape}')

        # Split the QKV into the heads
        Q = Q.view(T, self.heads, self.d_k)
        K = K.view(T, self.heads, self.d_k)
        V = V.view(T, self.heads, self.d_k)
        print(f'Shape of QKF after the split: {Q.shape}, {K.shape}, {V.shape}')

        # Tranpose to get the correct dim in their correct places
        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)
        print(f'Shape of QKF after the transpose: {Q.shape}, {K.shape}, {V.shape}')

        # Scores
        scores = Q @ K.transpose(-2, -1)
        print(f'Shape of scores: {scores.shape}')

        # Trill
        mask = torch.tril(torch.ones(T, T))
        scores_trill = scores.masked_fill(mask == 0, float('-inf'))
        print(f'Shape of scores trill: {scores_trill.shape}')

        # Attn
        attn = F.softmax(scores_trill, dim = -1)
        print(f'Shape of ATTN: {attn.shape}')

        # New Values
        new_values = attn @ V
        print(f'Shape of new_values: {new_values.shape}')

        # Go back to the (5,20)
        new_values = new_values.transpose(0, 1)
        new_values = new_values.contiguous().view(T, C)
        new_values = new_values @ self.Wo
        print(f'Shape of the output of transformers: {new_values.shape}')
        
        # ADD e NORM 1
        x = new_values + x
        x = self.norm1(x)
        print(f'Shape of first ADD e NORM (1): {x.shape}')

        # Go to the feedforward
        ffn1 = x @ self.W1 + self.bias
        print(f'Shape of first FFN (1): {ffn1.shape}')
        relu = F.relu(ffn1)

        ffn2 = relu @ self.W2 + self.bias2
        print(f'Shape of first FFN (1): {ffn2.shape}')

        # ADD e NORM 2 
        x = ffn2 + x
        x = self.norm2(x)
        print(f'Shape of second ADD e NORM (2): {x.shape}')

        return x

model = DecoderMHA()
output = model.forward(x = phrase)

print(f'==' * 50)
print(f'=' * 50)
print(f'\n\n\nOriginal Matrix (Embedding):\n{phrase}')
print(f'\nVS\n')
print(f'New Values Matrix (After Attn):\n{output}') 