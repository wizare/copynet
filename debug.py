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
        embed_x = x.clone()
        embed_x[embed_x>=self.vocab_size] = 2

        embedding = self.embed(embed_x)
        out ,h = self.gru(embedding)
        # print('encoder_outputs: ',out.shape)
        return out ,h




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
        self.attn_combine = nn.Linear(self.hidden_size*2+self.embed_size,self.hidden_size)
        self.attn_W = nn.Linear(self.hidden_size , self.hidden_size*2)
        # state rnn
        self.gru = nn.GRU(
            input_size=hidden_size*3,
            hidden_size=hidden_size,
        )

        #   weight 
        self.Wo = nn.Linear(hidden_size, vocab_size) # generate mode
        self.Wc = nn.Linear(hidden_size*2, hidden_size) # copy mode
        self.nonlinear = nn.Tanh()

    def forward(self,input_idx, encoder_outputs , input_seq_idx ,pre_state ,pre_prob_c ):
        #   input_idx       上一个预测输出的词的idx  [batch x 1]
        #   encoder_outputs encoder隐藏层矩阵        [ batch x input_seq_size x (hidden_size*2) ]
        #   input_seq_idx  输入序列的索引           [ batch x input_seq_size]
        #   pre_state       上一个隐藏层状态        [ batch x hidden_size]
        # print('input_idx:' , input_idx.shape)

        hidden_size = self.hidden_size
        vocab_size = self.vocab_size
        batch_size = encoder_outputs.size(0)
        input_seq_size = encoder_outputs.size(1)

        embed_x = input_idx.clone()
        embed_x[embed_x>=self.vocab_size] = 2
        embedding = self.embed(embed_x)
        # print('embedding: ',embedding.shape)
        #   reading encoder hidden
        
        #   attentive reading

        # attn_strength = self.attn(torch.cat((embedding.view(batch_size,-1) , pre_state ), 1))
        # print('attn_strength : ',attn_strength.shape)
        # attn_weights = F.softmax(attn_strength ,dim=1   ).view(batch_size,1,-1)
        # print('attn_weights: ',attn_weights.shape)

        attn_strength = self.attn_W(pre_state ).view(batch_size, -1, 1)
        # print('attn_strength' , attn_strength.shape)
        attn_scores = torch.bmm(encoder_outputs, attn_strength)
        # print('attn_scores: ',attn_scores.shape)
        attn_weights = F.softmax(attn_scores, dim=1).view(batch_size,1,-1) 
        # print('attn_weights: ',attn_weights.shape)

        context = torch.bmm( attn_weights, encoder_outputs  )  #   batch x 1 x hidden*2
        # print('context: ',context.shape)
        # attention combine
        attn_reading = self.attn_combine( torch.cat( (context, embedding ),2 ).squeeze(1) ).unsqueeze(1)

        #   selective reading

        idx_from_input = []
        for b in range(batch_size):
            idx_from_input.append( [int(i==input_idx[b].data[0]) for i in input_seq_idx[b].data.numpy() ] )

        idx_from_input = torch.Tensor(np.array(idx_from_input, dtype=float)).view(batch_size,-1)
        idx_from_input = Variable(idx_from_input)
        
        # print('idx_from_input: ',idx_from_input)
        # print(idx_from_input.sum(dim=1) )
        #   等下再看
        # if idx_from_input.sum().data[0]>1:
        #     idx_from_input = idx_from_input/idx_from_input.sum().data[0]

        select_weights = idx_from_input * pre_prob_c
        select_reading = torch.bmm(  select_weights.unsqueeze(1) ,encoder_outputs)


        #   state update
        gru_input = torch.cat( [attn_reading,select_reading ] ,2 ).view(1,batch_size,-1)

        pre_state = pre_state.view( 1,batch_size,-1)
        state , _ = self.gru(gru_input , pre_state )
        state = state.squeeze(0)
        # print('state: ',state.shape )
        

        #   predict

        #   generate mode
        score_g = self.Wo(state)  
        # print('score_g: ',score_g.shape)

        #   copy mode
        score_c = F.tanh(self.Wc(encoder_outputs.view(-1,hidden_size*2))) 
        score_c = score_c.view(batch_size,-1,hidden_size) 
        # print(score_c, state.unsqueeze(2))
        score_c = torch.bmm(score_c, state.unsqueeze(2)).view(batch_size,-1)
        # print( 'score_c: ',score_c.shape)

        #   softmax prob
        score = torch.cat([score_g,score_c],1) 
        prob = F.softmax(score ,dim=1)
        prob_g = prob[:,:vocab_size]    
        prob_c = prob[:,vocab_size:]    

        #   sum the same copy prob
        idx_from_input2 = []
        for b in range(batch_size):
            idx_from_input2.append( [ (input_seq_idx[b].data.numpy()==i)*1.0 for i in input_seq_idx[b].data.numpy() ])
        idx_from_input2 = np.array(idx_from_input2)
        idx_from_input2 = Variable( torch.Tensor(idx_from_input2) )

        prob_c =  torch.bmm( prob_c.unsqueeze(1) , idx_from_input2 ) # batch x 1 x input_seq_sizeq
        prob_c = prob_c.squeeze(1)                                   # batch x input_seq_sizeq

        # print('prob_g',prob_g.shape)
        # print('prob_c',prob_c.shape)
        #   生成最终输出
        out = torch.zeros(batch_size,self.max_length)
        for b in range(batch_size):
            for i,idx in enumerate(input_seq_idx[b].data.numpy()):
                if out[b][idx] == 0 and idx < self.max_length:
                    out[b][idx] = prob_c[b][i]
        out = Variable(out)
        if self.max_length>vocab_size:
            padding = Variable(torch.zeros(batch_size,self.max_length-vocab_size) )
            out += torch.cat([prob_g , padding  ], 1)
        else:
            out += prob_g

        # print('out: ',out)
        return out , state , prob_c





vocab_size = 12
embed_size = 12
hidden_size = 150
max_length = 12


encoder = Encoder(vocab_size, embed_size, hidden_size)
decoder = Decoder(vocab_size,embed_size,hidden_size,1, max_length)


x = torch.LongTensor([
    [1,3, 4, 5, 6, 7, 8, 9, 10, 11] ,
    [1,3, 4, 5, 6, 7, 8, 9, 10, 11] ,

])
# target = torch.LongTensor([ [3, 4, 5, 6, 7, 8, 9, 10, 11]  ])

target = x.clone()
target_idx = target[:,0]
print(target_idx.shape)
x = Variable(x)

def train(encoder,decoder,x,target,learning_rate=0.01):
    #
    batch_size = x.shape[0]
    input_seq_size = x.shape[1]
    hidden_size = decoder.hidden_size
    target_size = target.shape[1]

    from torch import optim
    loss = 0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss_func = nn.NLLLoss()

    input_idx = Variable( torch.ones(batch_size,dtype=torch.long).view(-1,1) )
    pre_state = Variable( torch.randn( batch_size,hidden_size ) )
    pre_prob_c = Variable( torch.zeros(batch_size,input_seq_size) )
    output_record = Variable( torch.ones(batch_size,dtype=torch.long).view(-1,1) )
    for i in range(target_size):

        target_idx = target[:,i]
        print(target_idx.shape)
        encoder_outputs , _ = encoder(x)

        decoder_output , pre_state , pre_prob_c = decoder(input_idx, encoder_outputs , x ,pre_state ,pre_prob_c )
        _ , input_idx = decoder_output.topk(1)
        input_idx = input_idx.detach()
        output_record = torch.cat( (output_record,input_idx ) ,dim=1 )
        print(decoder_output.shape)

        print('decoder_output: ',decoder_output.shape)
        print('target_idx: ',target_idx.shape)
        loss += loss_func(decoder_output, target_idx)

    output_record = output_record[:,1:]
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    print(output_record)
    print('loss: ', loss.item())
    print('-'*10)



for i in range(2):
    train(encoder,decoder,x,target)

