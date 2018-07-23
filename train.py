from encoder import Encoder
from decoder import Decoder

# 读取输入

encoder = Encoder(vocab_size , embed_size , hidden_size)
decoder = Decoder(vocab_size , embed_size  , hidden_size)
 

def train(input_idx, target_idx, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

	target_length = target_idx.size(0)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    encoder_outputs , _ = encoder(input_idx)

    decoder_input ,state , prob_c = decoder_initial()

    for y in range(target_length):
		decoder_output, state, prob_c = decoder(
            input_idx=decoder_input, 
            encoder_outputs=encoder_outputs,
	        input_seq_idx=input_out, 
            prev_state=state,
            pre_prob_c=prob_c,
        )
        top_word, top_idx = decoder_output.topk(1)
        decoder_input = top_idx.squeeze().detach() 
        loss += criterion(decoder_output, target_tensor[di])
        # 判断是否输出EOS    
        # if decoder_input = EOS
		
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder,decoder,epochs,learning_rate=0.01):


    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)


    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        #   获取input_idx , target_idx

        loss = train(input_idx, target_idx, 
            encoder, decoder,
             encoder_optimizer, decoder_optimizer, 
             criterion)