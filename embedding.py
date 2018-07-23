import torch
import torch.nn as nn
from torch.autograd import Variable

word2id = {'hello': 0, 'world': 1}
# you have 2 words, and then need 5 dim each word
embeds = nn.Embedding(4, 5)


hello_idx = torch.LongTensor([0])
print(hello_idx)

hello_idx = Variable(hello_idx)
print(hello_idx)


hello_embed = embeds(hello_idx)
print(hello_embed)