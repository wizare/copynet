import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self,vocab_size, embed_size, hidden_size):
        super(Encoder,self).__init__()

        self.embed = nn.Embedding(vocab_size,embed_size)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size = hidden_size,
            bidirectional=True
        )

    def forward(self,x):
        #   把 大于vocab_size 记为OOV
        # if x > self.vocab_size:
        #   x = 3

        x[x>self.vocab_size] = 3
        embedding = self.embed(x)
        print('embedding',embedding)
        out ,h = self.gru(embedding)
        return out ,h

encoder = Encoder(20,10,15)



x = torch.LongTensor([0,2,5,50,100]).unsqueeze(0)
x = Variable(x)
o , hidden = encoder(x)


