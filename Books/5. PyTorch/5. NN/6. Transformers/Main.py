# No PyTorch, você tem basicamente três níveis de dificuldade para implementar um Transformer
# A escolha depende de quanto você quer "sujar as mãos" com o cálculo matricial ou se quer apenas que a coisa funcione.
import torch 
import torch.nn as nn


# OPTION 1: The LEGO (High Level)
# 1.1
"""
    d_model = Dimensions of Embedding
    nhead = Number of heads per block
    dim_feedforward = Number of 'output' neurons
    num_x_layers = Number of blocks
    activation = Activation function after the ff

    Have the probability to use encoder + decoder in a single function
"""
encoder_decoder = nn.Transformer()




# 1.2
"""
    Here we have the encoder and decoder alone, but with the possibility of blocks
    x_layer = Define the block pattern for all the blocks
"""
encoder_block = nn.TransformerEncoder()
decoder_block = nn.TransformerDecoder()





# 1.3
"""
    Here we have the encoder and decoder in their pure form
"""
encoder = nn.TransformerEncoderLayer()
decoder = nn.TransformerDecoderLayer()





# OPTION 2: The Creator (Low Level)

"""
Here the MHA only gonna do the attention, and return the V values, without the ADD & NORM and the FeedForward

"""
mha = nn.MultiheadAttention()