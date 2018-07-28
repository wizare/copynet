import random

import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size ,  hidden_size):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, input, hidden):
        batch_length = input.size(0)
        embed_x = input.clone()
        embed_x[embed_x>=self.vocab_size] = 2

        embedded = self.embedding(embed_x).view(1, batch_length, self.embed_size)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self , batch_length):
        return torch.zeros(1, batch_length , self.hidden_size, device=device)

