import torch.nn as nn
import torch
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        """
        :param: input_dim is the size/dimensionality of the one-hot vectors that will be input to the encoder. This is equal to the input (source) vocabulary size.
        :param: emb_dim is the dimensionality of the embedding layer. This layer converts the one-hot vectors into dense vectors with emb_dim dimensions.
        :param: hid_dim is the dimensionality of the hidden and cell states.
        :param: n_layers is the number of layers in the RNN.
        :param: percentage of the dropout to use
        
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)
 
        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, num_layers=self.n_layers)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        :param: src sentences (src_len x batch_size)
        """
        # embedded = <TODO> (src_len x batch_size x embd_dim)
        embedded = self.embedding(src)
        # dropout over embedding
        embedded = self.dropout(embedded)
        outputs, (hidden, cell) = self.rnn(embedded)
        # [Attention return is for lstm, but you can also use gru]
        return outputs, hidden, cell
    
    
class Attention(nn.Module):
    def __init__(self, batch_size, hidden_dim_enc, hidden_dim_dec, method="dot"): # add parameters needed for your type of attention
        super(Attention, self).__init__()
        self.method = method # attention method you'll use. e.g. "cat", "one-layer-net", "dot", ...
        if method == 'general':
            self.attn_weights = nn.Linear(hidden_dim_dec, hidden_dim_enc)
    def forward(self, last_hidden, encoder_outputs, seq_len=None):
            '''
            encoder_outputs: seq_len, batch,  hidden_size_enc
            last_hidden(output nn.LSTM): 1, batch,  hidden_size_dec
            num_directions_dec == 1
            hidden_size_enc ==  hidden_size_dec for dot product
            '''
            if self.method == 'dot':
                scores = torch.sum(encoder_outputs * last_hidden, dim=-1) # seq_len, batch
            elif self.method == 'general':
                scores = self.attn_weights(last_hidden) #1, batch,  hidden_size_enc
                scores = torch.sum(encoder_outputs * last_hidden, dim=-1)# seq_len, batch
                
            scores = F.softmax(scores, dim=0)# seq_len, batch
            attention = encoder_outputs * scores.unsqueeze(-1)#seq_len, batch, hidden_size_enc
            attention = torch.sum(attention, dim=0, keepdim=True) #1,batch,  hidden_size_enc
            return attention
      
class DecoderAttn(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, attention, dropout=0.1):
        super(DecoderAttn, self).__init__()
        
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        self.attn = attention # instance of Attention class

        # define layers
        self.embedding = nn.Embedding(self.output_dim, self.emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=self.n_layers) #(lstm embd, hid, layers, dropout)
        self.out = nn.Linear(self.hid_dim*2, self.output_dim)
        self.dropout = nn.Dropout(dropout)

        # more layers you'll need for attention
        #<YOUR CODE HERE>
        
    def forward(self, input_, hidden, cell, encoder_output):
        # make decoder with attention
        # use code from seminar notebook as base and add attention to it
       # (1x batch_size)
        input_ = input_.unsqueeze(0)
        
        # (1 x batch_size x emb_dim)
        embedded = self.embedding(input_)# embd over input and dropout 
        embedded = self.dropout(embedded)
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        attn = self.attn(output, encoder_output) #1, batch, self.hid_dim
        
        concat = torch.cat([output, attn], dim=-1 )
        #sent len and n directions will always be 1 in the decoder
        
        # (batch_size x output_dim)
        
        prediction = self.out(concat.squeeze(0)) #project out of the rnn on the output dim 
        
        
        return prediction, hidden, cell

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        # Hidden dimensions of encoder and decoder must be equal
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self._init_weights() 
        self.max_len=30
    
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        """
        :param: src (src_len x batch_size)
        :param: tgt
        :param: teacher_forcing_ration : if 0.5 then every second token is the ground truth input
        """
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_out, hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0, :]
        
        
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell, enc_out) #TODO pass state and input throw decoder 
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
    
    def translate(self, src, TRG, device):
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = []
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        src = torch.tensor(src).to(self.device)
        enc_out, hidden, cell = self.encoder(src) # TODO pass src throw encoder
        
        #first input to the decoder is the <sos> tokens
        input = torch.tensor([TRG.vocab['<sos>']]).to(device)# TODO trg[idxs]
        
        for t in range(1, self.max_len):
            output, hidden, cell = self.decoder(input, hidden, cell, enc_out) #TODO pass state and input throw decoder 
            top1 = output.max(1)[1]
            outputs.append(top1)
            input = (top1)
        
        return outputs
    
    def _init_weights(self):
        p = 0.08
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -p, p)
        

