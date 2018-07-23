from torch import nn,optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch

class Decoder(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,max_length):
        super(Decoder,self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.max_length = max_length


        #attention mechanism
        self.attn = nn.Linear(self.hidden_size*2,max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2,self.hidden_size)

        # state rnn
        self.gru = nn.GRU(
            input_size=embed_size+hidden_size*2,
            hidden_size=hidden_size,
        )



        #   weight 
        self.Wo = nn.Linear(hidden_size, vocab_size) # generate mode
        self.Wc = nn.Linear(hidden_size*2, hidden_size) # copy mode
        self.nonlinear = nn.Tanh()

    def forward(self,input_idx, encoder_outputs , input_input_seq_idx_idx ,pre_state ,pre_prob_c ):
        #   input_idx       上一个预测输出的词的idx  1
        #   encoder_outputs encoder隐藏层矩阵        [ input_seq_size * (hidden_size*2) ]
        #   input_seq_idx  输入序列的索引           [ input_seq_size]
        #   pre_state       上一个隐藏层状态          [hidden_size]

        hidden_size = self.hidden_size

        vocab_size = self.vocab_size



        #   reading encoder hidden
        
        #   attentive reading

        attn_strength = self.attn(torch.cat((self.embed(input_idx) , pre_state ), 1))
        attn_weights = F.softmax(attn_strength , dim=1  )
        attn_applied = torch.bmm( attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)  )


        #   selective reading

        idx_from_input = [int(i==input_idx.data[0]) for i in input_seq_idx ]

        idx_from_input = torch.Tensor(np.array(idx_from_input, dtype=float))
        
 
        for i in range(b):
            if idx_from_input[i].sum().data[0]>1:
                idx_from_input[i] = idx_from_input[i]/idx_from_input[i].sum().data[0]

        select_weights = idx_from_input * pre_prob_c
        select_applied = select_weights * encoder_outputs
       

        #   state update
        gru_input = torch.cat( [attn_applied,select_applied ] ,1 )
        state = self.gru(gru_input , pre_state)


        #   predict

        #   generate mode
        score_g = self.Wo(state)   
        #   copy mode
        score_c = F.tanh(self.Wc(encoder_outputs.contiguous().view(-1,hidden_size*2))) 
        score_c = score_c.view(batch,-1,hidden_size) 
        score_c = torch.bmm(score_c, state.unsqueeze(2)).squeeze() 

        #   softmax prob
        score = torch.cat([score_g,score_c],1) 
        prob = F.softmax(score)
        prob_g = prob[:,:vocab_size]    
        prob_c = prob[:,vocab_size:]    



        #   生成最终输出
        out = torch.zeros(max_length)
        for i,idx in enumerate(input_input_seq_idx_idx):
            idx = int(idx)
            out[idx] += prob_c[i]
        out += torch.cat([prob_g,torch.zeros(max_length-vocab_size)],0)


        return out , state , prob_c