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
        out ,h = self.gru(embedding)
        return out ,h

vocab_size = 20
embed_size = 10
hidden_size = 15

encoder = Encoder(vocab_size, embed_size, hidden_size)

x = torch.LongTensor([0,2,5,50,100]).unsqueeze(0)
x = Variable(x)
encoder_outputs , hidden = encoder(x)


import torch.nn.functional as F
class Decoder(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,input_seq_size,max_length):
        super(Decoder,self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.max_length = max_length
        self.embed_size = embed_size
        self.input_seq_size = input_seq_size

        #attention mechanism
        self.attn = nn.Linear(self.hidden_size+self.embed_size ,self.input_seq_size )
        self.attn_combine = nn.Linear(self.hidden_size*2,self.hidden_size)

        # state rnn
        self.gru = nn.GRU(
            input_size=hidden_size*4,
            hidden_size=hidden_size,
        )

        #   weight 
        self.Wo = nn.Linear(hidden_size, vocab_size) # generate mode
        self.Wc = nn.Linear(hidden_size*2, hidden_size) # copy mode
        self.nonlinear = nn.Tanh()

    def forward(self,input_idx, encoder_outputs , input_seq_idx ,pre_state ,pre_prob_c ):
        #   input_idx       上一个预测输出的词的idx  1
        #   encoder_outputs encoder隐藏层矩阵        [ input_seq_size * (hidden_size*2) ]
        #   input_seq_idx  输入序列的索引           [ input_seq_size]
        #   pre_state       上一个隐藏层状态          [hidden_size]

        hidden_size = self.hidden_size

        vocab_size = self.vocab_size



        #   reading encoder hidden
        
        #   attentive reading

        attn_strength = self.attn(torch.cat((self.embed(input_idx) , pre_state.squeeze(0) ), 1))
        attn_weights = F.softmax(attn_strength   )
        
        attn_applied = torch.bmm( attn_weights.unsqueeze(0), encoder_outputs  )
        # attention combine


        #   selective reading


        idx_from_input = [int(i==input_idx.data[0]) for i in input_seq_idx.data.numpy() ]

        idx_from_input = torch.Tensor(np.array(idx_from_input, dtype=float)).view(1,-1)
        idx_from_input = Variable(idx_from_input)

        if idx_from_input.sum().data[0]>1:
            idx_from_input = idx_from_input/idx_from_input.sum().data[0]

        select_weights = idx_from_input * pre_prob_c
        select_applied = torch.bmm(  select_weights.unsqueeze(0) ,encoder_outputs)
        
        #   state update
        gru_input = torch.cat( [attn_applied,select_applied ] ,2 )
        _ , state = self.gru(gru_input , pre_state )
        state = state.squeeze(0)
        
        #   predict

        #   generate mode
        score_g = self.Wo(state)  
         
        #   copy mode
        score_c = F.tanh(self.Wc(encoder_outputs.view(-1,hidden_size*2))) 
        score_c = score_c.view(1,-1,hidden_size) 
        score_c = torch.bmm(score_c, state.unsqueeze(2)).view(1,-1)
       

        #   softmax prob
        score = torch.cat([score_g,score_c],1) 
        prob = F.softmax(score)
        prob_g = prob[:,:vocab_size]    
        prob_c = prob[:,vocab_size:]    




        idx_from_input2 = [ (input_seq_idx.data.numpy()==i)*1.0 for i in input_seq_idx.data.numpy() ]
        idx_from_input2 = np.array(idx_from_input2)
        # print(idx_from_input2)
        idx_from_input2 = Variable( torch.Tensor(idx_from_input2) )


        prob_c =  torch.mm( prob_c , idx_from_input2 )
        
         

        #   生成最终输出
        out = torch.zeros(self.max_length)
        for i,idx in enumerate(input_seq_idx.data.numpy()):
            if out[idx] != 0:
                out[idx] += prob_c.data.numpy()[i]
        out = Variable(out.view(1,-1))
        padding = Variable(torch.zeros(1,self.max_length-vocab_size) )

        out += torch.cat([prob_g , padding  ], 1)


        return out , state , prob_c




decoder = Decoder(vocab_size,embed_size,hidden_size,5, 30)
input_idx = Variable( torch.LongTensor([5]) )

input_seq_idx = Variable( torch.LongTensor([5,2,5,20,27]) )

pre_state = Variable( torch.randn(1,hidden_size ).unsqueeze(0) )

pre_prob_c = Variable( torch.randn(1,5 ) )

o , s , p = decoder(input_idx, encoder_outputs , input_seq_idx ,pre_state ,pre_prob_c )

print(o)