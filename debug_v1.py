'''

采用 attention 大致框架
autoencoder 的结果 还行
存在问题：
1.每个input_tensor 的输入长度需要固定为一致 
2.没有batch

'''

import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 15
embed_size = 10
hidden_size = 50
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size ,  hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



class AttnDecoderRNN(nn.Module):
    def __init__(self,vocab_size, embed_size, hidden_size , dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embed_size = embed_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        #   attn_reading
        self.attn = nn.Linear(self.hidden_size+self.embed_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size+self.embed_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(self.hidden_size*2, self.hidden_size)

        #   score
        self.Wo = nn.Linear(hidden_size, vocab_size) # generate mode
        self.Wc = nn.Linear(hidden_size, hidden_size) # copy mode

        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, input, hidden, encoder_outputs , input_seq , pre_prob):

        # print('input: ',input.item())
        # print('input_seq: ',input_seq)
        # print('pre_prob: ',pre_prob)
        sel_weights = (input.item() == input_seq).long() 
        sel_weights =  sel_weights * pre_prob 
        sel_weights = sel_weights.view(1,-1)
        sel_reading = torch.mm(sel_weights ,encoder_outputs.long()).unsqueeze(1).float()

        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        #   attn 
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)



        
        gru_input = torch.cat( (output,sel_reading) ,2)
        # print('gru_input: ',gru_input.shape)
        # output, hidden = self.gru(output, hidden)
        output, hidden = self.gru(gru_input, hidden)   #    output same as hidden
        # print(hidden.shape) # 1 * 1 * hidden

        score_g = self.Wo(hidden).view(1,-1)
        # print(score_g.shape)  #   1 * vocab_size

        score_c = self.Wc(encoder_outputs)
        score_c = F.tanh(score_c)
        score_c = torch.mm(score_c , hidden.view(-1,1) ).view(1,-1)
        # print(score_c.shape)    # 1 * seq_size



        score = F.log_softmax( torch.cat( (score_g,score_c) ,dim=1 ),dim=1 )
        prob_g = score[:,:vocab_size]   # 1 * vocab_size 
        prob_c = score[:,vocab_size:]    # 1 * seq_size

        
        output1 = torch.zeros(1,vocab_size)
        output1 += prob_g

        seq_size = input_seq.shape[0]
        one_hot = torch.zeros(seq_size,vocab_size).scatter_(1,input_seq.view(-1,1),1)

        output1 += torch.mm(prob_c , one_hot )

        return output1 , hidden , attn_weights , prob_c

        # output = F.log_softmax(self.out(output[0]), dim=1)
        # return output, hidden, attn_weights


    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    output_rec = []
    input_tensor_clone = input_tensor.clone()
    pre_prob = torch.zeros(input_length,dtype=torch.long)
    for di in range(target_length):
        # decoder_output, decoder_hidden, decoder_attention = decoder(
        #     decoder_input, 
        #     decoder_hidden, 
        #     encoder_outputs,
        #     input_tensor_clone,
        #     pre_prob
        # )
        decoder_output, decoder_hidden, decoder_attention , prob_c = decoder(
            decoder_input, 
            decoder_hidden, 
            encoder_outputs,
            input_tensor_clone,
            pre_prob
        )
        pre_prob = prob_c.squeeze().long().detach()  # detach from history as input

        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        output_rec.append(topi.item())
        loss += criterion(decoder_output, target_tensor[di].view(1))
        if decoder_input.item() == EOS_token:
            break

    loss.backward()
    print(output_rec)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length








input_tensor = torch.tensor( [3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.long)
target_tensor = torch.tensor( [3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.long)

MAX_LENGTH = input_tensor.shape[0]

encoder = EncoderRNN(vocab_size ,embed_size ,  hidden_size)
decoder = AttnDecoderRNN( vocab_size ,embed_size ,  hidden_size,max_length=MAX_LENGTH)

learning_rate=0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

criterion = nn.NLLLoss()    



for i in range(200):
	train(
		input_tensor, 
		target_tensor, 
		encoder, decoder, 
		encoder_optimizer, 
		decoder_optimizer, 
		criterion, 
		max_length=MAX_LENGTH
	)

