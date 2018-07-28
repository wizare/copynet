'''

添加batch操作 7.24
encoder bidirectional 7.25


'''

import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    batch_length = input_tensor.size(0)
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)

    encoder_hidden = encoder.initHidden(batch_length)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs = torch.zeros( batch_length, max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder( input_tensor[:,ei], encoder_hidden)
        encoder_outputs[:,ei] = encoder_output[:,0]

    # decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_input = torch.ones( batch_length ,dtype=torch.long , device=device) * SOS_token

    decoder_hidden = encoder_hidden

    output_rec = torch.zeros(batch_length,max_length )
    input_tensor_clone = input_tensor.clone()
    pre_prob = torch.zeros( batch_length , input_length,dtype=torch.long)
    for di in range(target_length):
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
        # print('decoder_input: ',decoder_input)
        output_rec[:,di] =  topi.view(-1)
        # output_rec.append(topi.item())

        loss += criterion(decoder_output, target_tensor[:,di])
        # if decoder_input.item() == EOS_token:
        #     break

    loss.backward()
    print(output_rec)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length



def evaluate(encoder, decoder, input_tensor, max_length):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        output_rec = []
        # decoder_attentions = torch.zeros(max_length, max_length)
        input_tensor_clone = input_tensor.clone()
        pre_prob = torch.zeros(input_length,dtype=torch.long)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention , prob_c = decoder(
                decoder_input, 
                decoder_hidden, 
                encoder_outputs,
                input_tensor_clone,
                pre_prob
            )
            pre_prob = prob_c.squeeze().long().detach()  # detach from history as input
            # decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                output_rec.append('<EOS>')
                break
            else:
                output_rec.append(topi.item())

            decoder_input = topi.squeeze().detach()
        print('evaluate------------------')
        print(output_rec)  
        # return decoded_words, decoder_attentions[:di + 1]



vocab_size = 25
max_vocab_size = 30
embed_size = 20
hidden_size = 70
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1


input_tensor = torch.tensor( 
[[  3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,   1.,   1.],
        [ 15.,   7.,   8.,   9.,  16.,   1.,   1.,   1.,   1.,   1.],
        [  3.,   4.,   5.,   6.,   7.,  19.,  20.,  10.,   1.,   1.]],
     dtype=torch.long)
target_tensor = torch.tensor( 
    [[ 11.,  12.,  13.,  14.,   4.,   8.,   9.,  10.,   1.,   1.],
        [  8.,   9.,   7.,  17.,  18.,   1.,   1.,   1.,   1.,   1.],
        [ 11.,  12.,  13.,  14.,   4.,  19.,  20.,   2.,   1.,   1.]],
    dtype=torch.long)

MAX_LENGTH = input_tensor.shape[1]

encoder = Encoder(vocab_size ,embed_size ,  hidden_size)
decoder = Decoder( vocab_size ,embed_size ,  hidden_size, max_vocab_size , max_length=MAX_LENGTH)

learning_rate=0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

criterion = nn.NLLLoss()    



for i in range(300):
	train(
		input_tensor, 
		target_tensor, 
		encoder, decoder, 
		encoder_optimizer, 
		decoder_optimizer, 
		criterion, 
		max_length=MAX_LENGTH
	)

# eva_x = torch.tensor( [3, 4, 5, 6, 7, 20, 21, 10, 11], dtype=torch.long)
# evaluate(encoder , decoder , eva_x , max_length=MAX_LENGTH)