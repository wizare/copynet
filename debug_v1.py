'''
采用 attention 大致框架
autoencoder 的结果 还行
存在问题：
1.每个input_tensor 的输入长度需要固定为一致 
2.没有batch   （一个batch中，一些句子输出eos终止 ， 一些应该继续训练 ， 
怎么处理？ 没想好

3. encoder 没有用rnn
4. oov词没有额外体现
'''

import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 18
max_vocab_size = 21
embed_size = 15
hidden_size = 50
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size ,  hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size,bidirectional=True)

    def forward(self, input, hidden):

        embed_x = input.clone()
        embed_x[embed_x>=self.vocab_size] = 2
        embedded = self.embedding(embed_x).view(1, 1, -1)

        # embedded = self.embedding(input).view(1, 1, -1)
        output = embedded                        
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)



class Decoder(nn.Module):
    def __init__(self,vocab_size, embed_size, hidden_size ,eo_hidden_size, max_vocab_size ,dropout_p=0.1, max_length=MAX_LENGTH):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.eo_hidden_size = eo_hidden_size
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.embed_size = embed_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        #   attn_reading
        self.attn = nn.Linear(self.hidden_size+self.embed_size, self.max_length)
        self.attn_combine = nn.Linear(self.eo_hidden_size+self.embed_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(self.hidden_size+self.eo_hidden_size, self.hidden_size)

        #   score
        self.Wo = nn.Linear(self.hidden_size, vocab_size) # generate mode
        self.Wc = nn.Linear(self.eo_hidden_size, hidden_size) # copy mode

        # self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, input, hidden, encoder_outputs , input_seq , pre_prob):

        # print('input: ',input.item())
        # print('input_seq: ',input_seq)
        # print('pre_prob: ',pre_prob)
        sel_weights = (input.item() == input_seq).long() 
        sel_weights =  sel_weights * pre_prob 
        sel_weights = sel_weights.view(1,-1)
        sel_reading = torch.mm(sel_weights ,encoder_outputs.long()).unsqueeze(1).float()

        embed_x = input.clone()
        embed_x[embed_x>=self.vocab_size] = 2
        embedded = self.embedding(embed_x).view(1, 1, -1)
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
        # print('hidden:',hidden.shape) # 1 * 1 * hidden

        score_g = self.Wo(hidden).view(1,-1)
        # print(score_g.shape)  #   1 * vocab_size

        score_c = self.Wc(encoder_outputs)
        score_c = F.tanh(score_c)
        score_c = torch.mm(score_c , hidden.view(-1,1) ).view(1,-1)
        # print(score_c.shape)    # 1 * seq_size



        score = F.softmax( torch.cat( (score_g,score_c) ,dim=1 ),dim=1 )
        prob_g = score[:,:vocab_size]   # 1 * vocab_size 
        prob_c = score[:,vocab_size:]    # 1 * seq_size

        # print('prob_c: ',prob_c)
        if self.max_vocab_size>vocab_size:
            padding = torch.zeros(1,self.max_vocab_size-vocab_size) 
            output1 = torch.cat([prob_g , padding  ], 1)
        else:
            output1 = prob_g

        seq_size = input_seq.shape[0]
        one_hot = torch.zeros(seq_size,max_vocab_size).scatter_(1,input_seq.view(-1,1),1)
        output1 += torch.mm(prob_c , one_hot )

        output1[output1==0] = 1e-9
        output1[0][2] = 1e-9

        output1 = output1.log()

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

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size*2, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder( input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = decoder.initHidden()

    output_rec = []
    input_tensor_clone = input_tensor.clone()
    pre_prob = torch.zeros(input_length,dtype=torch.long)
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
        output_rec.append(topi.item())
        loss += criterion(decoder_output, target_tensor[di].view(1))
        if decoder_input.item() == EOS_token:
            break

    loss.backward()
    print(output_rec)
    encoder_optimizer.step()
    decoder_optimizer.step()
    print('loss: ',loss.item())
    return loss.item() / target_length



def evaluate(encoder, decoder, input_tensor, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size*2, device=device)

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

def trainIter(input_tensors, target_tensors,n_iters, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    batch_length = len(input_tensors)
    for i in range(batch_length):
        input_tensor = input_tensors[i]
        target_tensor = target_tensors[i]

        for i in range(n_iters):
            train(
                input_tensor, 
                target_tensor, 
                encoder, decoder, 
                encoder_optimizer, 
                decoder_optimizer, 
                criterion, 
                max_length=MAX_LENGTH
            )

input_list = [ 
    [3, 4, 5, 6, 7, 19, 20, 10, 1, 1]
 ]
target_list = [ 
    [11, 12, 13, 14, 4, 19, 20]
]
input_tensors = [torch.tensor( i , dtype=torch.long) for i in input_list]
target_tensors = [ torch.tensor( i, dtype=torch.long) for i in target_list ] 


encoder = Encoder(vocab_size ,embed_size ,  hidden_size)
decoder = Decoder( vocab_size ,embed_size , hidden_size,hidden_size*2, max_vocab_size ,max_length=MAX_LENGTH)

learning_rate=0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

criterion = nn.NLLLoss()    

n_iters = 1000
trainIter(
        input_tensors, 
        target_tensors, 
        n_iters,
        encoder, decoder, 
        encoder_optimizer, 
        decoder_optimizer, 
        criterion, 
        max_length=MAX_LENGTH
    )

# eva_x = torch.tensor( [3, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
# evaluate(encoder , decoder ,eva_x , max_length=MAX_LENGTH)