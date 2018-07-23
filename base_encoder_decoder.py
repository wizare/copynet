import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class EncoderRNN(nn.Module):
	def __init__(self, vocab_size , embed_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(vocab_size , embed_size )
		self.gru = nn.GRU(embed_size, hidden_size)

	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1, 1, -1)

		output = embedded
		output, hidden = self.gru(output, hidden)
		return output, hidden
	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size)

class DecoderRNN(nn.Module):
	def __init__(self,vocab_size , embed_size, hidden_size):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.gru = nn.GRU(embed_size , hidden_size)
		self.out = nn.Linear(hidden_size, vocab_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):

		output = self.embedding(input)

		output = F.relu(output)

		output, hidden = self.gru(output, hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden


vocab_size = 15
embed_size = 10
hidden_size = 50

x = torch.tensor( [1,3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.long)


encoder = EncoderRNN( vocab_size , embed_size, hidden_size )
decoder = DecoderRNN( vocab_size , embed_size, hidden_size )

encoder_hidden = encoder.initHidden()

for i in range( x.shape[0] ):
	x_i = x[i]
	_ , encoder_hidden = encoder(  x_i , encoder_hidden  )



# x_i = torch.tensor( [[1]], dtype=torch.long).view(-1,1)
o , h = decoder(x_i , hidden)



