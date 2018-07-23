'''

尝试加batch ，
快完成时，觉得并不需要。
十分奇怪

'''

import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 25
embed_size = 20
hidden_size = 70
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size ,  hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, input, hidden):
        batch_length = input.size(0)
        embedded = self.embedding(input).view(1, batch_length, self.embed_size)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 2, self.hidden_size, device=device)



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
        batch_length = encoder_outputs.size(0)

        print('input: ',input.shape)

        sel_weights = torch.zeros(batch_length ,self.max_length , dtype=torch.long)
        for i in range(batch_length):
            sel_weights[i , :] = (input[i].item() == input_seq[i]).long() 
        sel_weights =  sel_weights * pre_prob 
        sel_weights = sel_weights.view(batch_length,1,-1)
        print('sel_weights: ' , sel_weights.shape)
        sel_reading = torch.bmm(sel_weights ,encoder_outputs.long()).float()
        print('sel_reading:' , sel_reading.shape)

        print(input)
        embedded = self.embedding(input).view(batch_length, 1, -1)
        embedded = self.dropout(embedded)

        print('embedded: ',embedded.shape)
        print('hidden:' , hidden.shape)
        #   attn 
        a = torch.cat((embedded.view(batch_length,-1), hidden.view(batch_length,-1)  ), 1)
        print(a.shape)
        a = self.attn(a)
        attn_weights = F.softmax( a , dim=1)
        print('attn_weights: ',attn_weights.shape)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        print('attn_applied: ',attn_applied.shape)
        output = torch.cat((embedded, attn_applied), 2).squeeze(1)
        output = self.attn_combine(output).unsqueeze(1)
        output = F.relu(output)



        print('output: ' ,output.shape)
        gru_input = torch.cat( (output,sel_reading) ,2).view(1,batch_length,-1)
        print('gru_input: ',gru_input.shape)
        # output, hidden = self.gru(output, hidden)
        output, hidden = self.gru(gru_input, hidden)   #    output same as hidden
        hidden = hidden.squeeze(0)
        print('hidden: ',hidden.shape) #  b * hidden

        score_g = self.Wo(hidden).view(batch_length,-1)
        print('score_g:',score_g.shape)  #   b * vocab_size

        score_c = self.Wc(encoder_outputs)
        score_c = F.tanh(score_c)
        score_c = torch.bmm(score_c , hidden.view(batch_length,-1 ,1) ).squeeze(-1)
        print('score_c: ',score_c.shape)    # b * seq_size



        score = F.log_softmax( torch.cat( (score_g,score_c) ,dim=1 ),dim=1 )
        prob_g = score[:,:vocab_size]   # b * vocab_size 
        prob_c = score[:,vocab_size:]    # b * seq_size

        
        output1 = torch.zeros(batch_length,vocab_size)
        output1 += prob_g

        seq_size = self.max_length
        one_hot = torch.zeros(batch_length,seq_size,vocab_size).scatter_(2,input_seq.view(batch_length,-1,1),1)
        print(one_hot.shape)
        output1 += torch.bmm(prob_c.unsqueeze(1) , one_hot ).squeeze(1)

        return output1 , hidden , attn_weights , prob_c

        # output = F.log_softmax(self.out(output[0]), dim=1)
        # return output, hidden, attn_weights


    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()


    batch_length = input_tensor.size(0)
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(0)

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
        print(decoder_input.)
        output_rec[:,di] =  topi.view(-1)
        # output_rec.append(topi.item())
        loss += criterion(decoder_output, target_tensor[di].view(1))
        if decoder_input.item() == EOS_token:
            break

    loss.backward()
    print(output_rec)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length



def evaluate(encoder, decoder, input_tensor, max_length=MAX_LENGTH):
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




input_tensor = torch.tensor( 
    [
        [3, 4, 5, 6, 7, 8, 9, 10, 11],
        [3, 4, 5, 6, 7, 20, 21, 10, 11],
    ],
     dtype=torch.long)
target_tensor = torch.tensor( [12, 13, 14, 15, 4, 8, 9, 10], dtype=torch.long)

MAX_LENGTH = input_tensor.shape[1]

encoder = EncoderRNN(vocab_size ,embed_size ,  hidden_size)
decoder = AttnDecoderRNN( vocab_size ,embed_size ,  hidden_size,max_length=MAX_LENGTH)

learning_rate=0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

criterion = nn.NLLLoss()    



for i in range(2):
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