import torch
import torch.nn.functional as F
import torch.nn as nn

"""
First we need to create our data, i'll just get a embedding (with positional too) phrase already done!

- 5 Words (tokens)
- The embedding have 10 dim size

"""
phrase = torch.rand(5, 10)

print(f'Phrase:\n{phrase}\n')

class DecoderSingleHeadBlock:
    """
    Now get down on the math:

    1. Create the QKV matrices
    2. Get the Q * K product (scores) 
    3. Get the trill of the scores
    4. Get the softmax of the scores now trill
    5. Multiplication the softmax (scores trill) for the V matrix and get the Nvalues
    6. Add the residual connection + NORM
    7. Linear part, normally we increase the dim in 4 time (with relu including)
    8. Add the residual connection + NORM
    9. The result will be the output of the block, and this will be pass for the next layer (but here we only gonna use 1 block/layer)
    
    Here I'm gonna use just: 
    1 head of attn
    1 block of transformer
    
    """


    def __init__(self):
        # Attn
        self.Wq = torch.rand(10, 10)
        self.Wv = torch.rand(10, 10)
        self.Wk = torch.rand(10, 10)

        # FFN
        self.W1 = torch.rand(10, 40)
        self.W2 = torch.rand(40, 10)
        self.bias = torch.rand(40)
        self.bias2 = torch.rand(10)

        # Norm
        self.norm1 = torch.nn.LayerNorm(10)
        self.norm2 = torch.nn.LayerNorm(10)

      

    def forward(self, x):

        # 1. Create the QKV Matrices
        Q = x @ self.Wq
        K = x @ self.Wk  
        V = x @ self.Wv 
        print(f'Q Matrice:\n {Q}, shape: {Q.shape}')
        print(f'K Matrice:\n {K}, shape: {K.shape}')
        print(f'V Matrice:\n {V}, shape: {V.shape}\n')


        # 2. Calculate the scores
        scores = Q @ K.T
        print(f'Scores: \n{scores}\n')

        # 3. Trill the scores to don't see the future
        scores_trill = torch.tril(scores)
        print(f'Scores trill:\n{scores_trill}')
        
        # 4. Get the softmax of the scores
        attn = F.softmax(scores_trill, dim = 1)
        print(f'Softmax:\n{attn}\n')

        # 5. Get the new values
        output_attention = attn @ V
        print(f'Output Attention: \n{output_attention}')

        # 6. ADD and NORM 1
        x = x + output_attention
        x = self.norm1(x)
        print(f'ADD & NORM 1:\n{x}\n') 

        # 7. Feed Forward 
        ffn1_output = x @ self.W1 + self.bias
        relu1_output = F.relu(ffn1_output)
        ffn2_output = relu1_output @ self.W2 + self.bias2
        print(f'After FFN and RELU: \n{ffn2_output}, shape: {ffn2_output.shape}') 

        # 8. ADD and NORM 2
        x = x + ffn2_output
        x = self.norm2(x)
        print(f'ADD & NORM 2:\n{x}\n') 

        return x
        

model = DecoderSingleHeadBlock()
output = model.forward(x=phrase)


torch.set_printoptions(
    precision=2,    
    sci_mode=False, 
    linewidth=100  
)
print(f'=' * 50)
print(f'\n\n\nOriginal Matrix (Embedding):\n{phrase}\n')
print(f'\nVS\n')
print(f'New Values Matrix (After Attn):\n{output}')