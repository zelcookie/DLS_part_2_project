import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
import math
from torch import Tensor
from typing import Optional, Any
import copy

import torch.nn.functional as F

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PruneMask(nn.Module):

    def __init__(self,
                 heads,
                 init_mean=0.5,
                 init_std=0.01,
                 beta=1.0,
                 stretch=0.1,
                 
                        ):
        super().__init__()

        self.heads =heads
        self.limit_l = -stretch
        self.limit_r = 1.0 + stretch
        self.log_alpha = nn.Parameter(torch.zeros(self.heads))
        self.beta = beta
        self.init_mean = init_mean
        self.init_std = init_std
        mean = math.log(1 - self.init_mean) - math.log(self.init_mean)
        self.log_alpha.data.normal_(mean, self.init_std)

    def l0_norm(self) :

        return (self.log_alpha - self.beta * math.log(-self.limit_l / self.limit_r)).sigmoid().sum()

    def forward(self):
        if self.training:

            u = self.log_alpha.new(self.heads).uniform_(0, 1)
            s = F.sigmoid((torch.log(u)- torch.log(1 - u) + self.log_alpha) / self.beta)
            #s = F.sigmoid(self.log_alpha / self.beta)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
            mask = s.clamp(min=0., max=1.)

        else:
            soft_mask = F.sigmoid(self.log_alpha / self.beta)
            mask = (soft_mask>0.5).type(torch.long)
        #print(self.log_alpha)
        return mask



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

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))    
    
    
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn



class MultiheadAttentionPrune(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiheadAttentionPrune, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.prune_mask = PruneMask(h)
        self.use_prune_mask=False
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        if self.use_prune_mask:
            
            
            x = x * self.prune_mask().reshape(1,-1,1,1)
#         print(x)
#         print(self.prune_mask())
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
#         print(x)
#         print(self.prune_mask())
#         print(self.linears[-1](x))
        return self.linears[-1](x)
    
    
class TransformerEncoderLayerPrune(nn.Module):
 

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayerPrune, self).__init__()
        self.self_attn = MultiheadAttentionPrune(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayerPrune, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        src2 = self.self_attn(src, src, src)
        #print(src2.shape)
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
class EncoderPrune(nn.Module):
    def __init__(self, input_dim, emb_dim, n_layers, heads, dropout=0.1):
        super().__init__()
        self.emb_dim=emb_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.pe = PositionalEncoding(emb_dim, dropout)
        
        
        encoder_layer =  TransformerEncoderLayerPrune(emb_dim, heads, dropout=dropout)
        norm = nn.LayerNorm(emb_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers, norm)
        self.set_prune_mask_flag(False)

    def forward(self, src, src_key_padding_mask):
        embedded = self.embedding(src)

        embedded = self.pe(embedded*math.sqrt(self.emb_dim))

        out = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        #print(out[:,:,2])
        return out  
   
    def set_prune_mask_flag(self, flag):
        for i, layer in enumerate(self.transformer_encoder.layers):
            if i!=0:
                layer.self_attn.use_prune_mask = flag
            
    def prune_loss(self):
        loss = 0
        for layer in self.transformer_encoder.layers:
            
            if layer.self_attn.use_prune_mask:
                loss += layer.self_attn.prune_mask.l0_norm()
        return loss
    
    
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, n_layers, heads, dropout=0.1):
        super().__init__()
        self.emb_dim=emb_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.pe = PositionalEncoding(emb_dim, dropout)
        encoder_layer =  TransformerEncoderLayer(emb_dim, heads, dropout=dropout)
        norm = nn.LayerNorm(emb_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers, norm)

    def forward(self, src, src_key_padding_mask):
        embedded = self.embedding(src)
        #print(embedded.shape)
        embedded = self.pe(embedded*math.sqrt(self.emb_dim))
        #print(embedded)
        out = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        #print(out[:,:,2])
        return out


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, n_layers, heads, dropout=0.1):
        super().__init__()
        self.emb_dim=emb_dim
        self.output_dim = output_dim 
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.pe = PositionalEncoding(emb_dim, dropout)
        decoder_layer = TransformerDecoderLayer(emb_dim, heads, dropout=dropout)
        norm = nn.LayerNorm(emb_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, n_layers, norm)
        self.out = nn.Linear(emb_dim, self.output_dim)

    def forward(self, trg, encoder_out,tgt_key_padding_mask, tgt_mask):
        embedded = self.embedding(trg)
        #print(embedded.shape)
        embedded = self.pe(embedded*math.sqrt(self.emb_dim))
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
    
    def forward(self, src, trg,src_key_padding_mask, tgt_key_padding_mask, tgt_mask, teacher_forcing_ratio = 0.5, return_prune_loss=False):
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
        if return_prune_loss:
            prune_loss = self.encoder.prune_loss()
            return outputs, prune_loss
        return outputs
    