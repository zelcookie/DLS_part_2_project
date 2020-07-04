import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
import math



class PositionalEncoding(nn.Module):

    def __init__(self, emb_dim, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    
    

    
    
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, n_layers, heads, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.pe = PositionalEncoding(emb_dim, dropout)
        encoder_layer =  TransformerEncoderLayer(emb_dim, heads, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers)

    def forward(self, src, src_key_padding_mask):
        embedded = self.embedding(src)
        #print(embedded.shape)
        embedded = self.pe(embedded)
        #print(embedded)
        out = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        #print(out[:,:,2])
        return out


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, n_layers, heads, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim 
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.pe = PositionalEncoding(emb_dim, dropout)
        decoder_layer = TransformerDecoderLayer(emb_dim, heads, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, n_layers)
        self.out = nn.Linear(emb_dim, self.output_dim)

    def forward(self, trg, encoder_out,tgt_key_padding_mask, tgt_mask):
        embedded = self.embedding(trg)
        #print(embedded.shape)
        embedded = self.pe(embedded)
        #print(embedded.shape, encoder_out.shape, tgt_mask.shape,tgt_key_padding_mask.shape )
        
        out = self.transformer_decoder(embedded, encoder_out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        out = self.out(out)
        return out


    
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        # Hidden dimensions of encoder and decoder must be equal
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        #self._init_weights() 
        self.max_len=30
    
    def forward(self, src, trg,src_key_padding_mask, tgt_key_padding_mask, tgt_mask, teacher_forcing_ratio = 0.5):
        """
        :param: src (src_len x batch_size)
        :param: tgt
        :param: teacher_forcing_ration : if 0.5 then every second token is the ground truth input
        """
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_out = self.encoder(src, src_key_padding_mask)
        #print(enc_out)
        #first input to the decoder is the <sos> tokens
#         input = trg
        
# #         for t in range(1, max_len):
# #             output = self.decoder(input, enc_out, tgt_mask) #TODO pass state and input throw decoder 
# #             outputs[t] = output
# #             teacher_force = random.random() < teacher_forcing_ratio
# #             top1 = output.max(1)[1]
# #             input = (trg[t] if teacher_force else top1)
        
        outputs = self.decoder(trg,  enc_out,tgt_key_padding_mask, tgt_mask)
        
        return outputs
    
    def translate(self, src):
        
        enc_out = self.encoder(src)
        
        
        
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
    
    
    
    
    
    
    
    
    
    
    
    


