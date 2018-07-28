import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Decoder(nn.Module):
    def __init__(self,vocab_size, embed_size, hidden_size , max_vocab_size ,   max_length , dropout_p=0.1,):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.max_vocab_size = max_vocab_size
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
        

        # print('input: ',input.shape)

        sel_weights = torch.zeros(batch_length ,self.max_length , dtype=torch.long)
        for i in range(batch_length):
            sel_weights[i , :] = (input[i].item() == input_seq[i]).long() 
        sel_weights =  sel_weights * pre_prob 
        sel_weights = sel_weights.view(batch_length,1,-1)
        # print('sel_weights: ' , sel_weights.shape)
        sel_reading = torch.bmm(sel_weights ,encoder_outputs.long()).float()
        # print('sel_reading:' , sel_reading.shape)



        embedded = self.embedding(input).view(batch_length, 1, -1)
        embedded = self.dropout(embedded)

        # print('embedded: ',embedded.shape)
        # print('hidden:' , hidden.shape)
        #   attn 
        a = torch.cat((embedded.view(batch_length,-1), hidden.view(batch_length,-1)  ), 1)
        # print(a.shape)
        a = self.attn(a)
        attn_weights = F.softmax( a , dim=1)
        # print('attn_weights: ',attn_weights.shape)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        # print('attn_applied: ',attn_applied.shape)
        output = torch.cat((embedded, attn_applied), 2).squeeze(1)
        output = self.attn_combine(output).unsqueeze(1)
        output = F.relu(output)

        # print('output: ' ,output.shape)
        gru_input = torch.cat( (output,sel_reading) ,2).view(1,batch_length,-1)
        # print('gru_input: ',gru_input.shape)
        # output, hidden = self.gru(output, hidden)
        output, hidden = self.gru(gru_input, hidden)   #    output same as hidden
        hidden = hidden.squeeze(0)
        # print('hidden: ',hidden.shape) #  b * hidden

        score_g = self.Wo(hidden).view(batch_length,-1)
        # print('score_g:',score_g.shape)  #   b * vocab_size

        score_c = self.Wc(encoder_outputs)
        score_c = F.tanh(score_c)
        score_c = torch.bmm(score_c , hidden.view(batch_length,-1 ,1) ).squeeze(-1)
        # print('score_c: ',score_c.shape)    # b * seq_size



        score = F.softmax( torch.cat( (score_g,score_c) ,dim=1 ),dim=1 )
        prob_g = score[:,:self.vocab_size]   # b * vocab_size 
        prob_c = score[:,self.vocab_size:]    # b * seq_size

        
        if self.max_vocab_size>self.vocab_size:
            padding = torch.zeros(batch_length,self.max_vocab_size-self.vocab_size) 
            output1 = torch.cat([prob_g , padding  ], 1)
        else:
            output1 = prob_g


        seq_size = self.max_length
        one_hot = torch.zeros(batch_length,seq_size,self.max_vocab_size).scatter_(2,input_seq.view(batch_length,-1,1),1)
        
        output1 += torch.bmm(prob_c.unsqueeze(1) , one_hot ).squeeze(1)
        hidden = hidden.unsqueeze(0)
        output1[output1==0] = 1e-9
        output1[:,2] = 1e-9

        output1 = output1.log()
        # print('output1: ',output1)
        return output1 , hidden , attn_weights , prob_c

        # output = F.log_softmax(self.out(output[0]), dim=1)
        # return output, hidden, attn_weights


