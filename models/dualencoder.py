import torch
import torch.nn as nn


class Encoder(nn.Module):
    
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout, vocab, use_pretrained=True):
        
        """RNN Encoder"""
        
        super().__init__()
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True) # bidirectional en argument
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim) 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src_emb):
        
        """
        Encoding input sequence with RNN layer
        Input : src_emb - embedded input sequence - [src len, batch size, emb_dim]
        Output : encoder_outputs - encoded output for input - 
                 encoder_hidden - hidden layer for input - 
        """
        embedded = self.dropout(src_emb)                          # [src len, batch size, emb dim]
        
        # encoder_outputs = [src len, batch size, enc hid dim * num directions]
        # encoder_hidden = [n layers * num directions, batch size, hid dim]
        encoder_outputs, encoder_hidden = self.rnn(embedded)
            
        # initial decoder hidden is final encoder hidden state of the forwards and backwards RNNs fed through a linear layer
        # encoder_hidden = [batch size, dec hid dim]
        encoder_hidden = torch.tanh(self.fc(torch.cat((encoder_hidden[-2,:,:], encoder_hidden[-1,:,:]), dim = 1))) # encoder hidden[-2, :, :] = [batch size, hid dim]
        
        return encoder_outputs, encoder_hidden
    
    
class DualEncoder(nn.Module):
    
    """Create encoding representation for original text and previous iteration and generated text"""
    
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout, vocab, use_pretrained=True): 
                
        super().__init__()
        
        self.encoder = Encoder(emb_dim=emb_dim, 
                               enc_hid_dim=enc_hid_dim,
                               dec_hid_dim=dec_hid_dim,
                               dropout=dropout,
                               vocab=vocab,
                               use_pretrained=use_pretrained)
        
    def forward(self, src_emb, prev_emb):
        
        """
        Concatenate encoded represetation for current and previously generated text
        Input : src_emb - embedded input sequence - [src len, batch size, emb_dim]
                prev_emb - embedd sequence of previously generated text - [prev_gen_text, batch_size, emb_dim]
        Output : stacked_outputs - concatenante encoded output for both inputs
                 stacked_hidden - concatenante hidden layer for both inputs
        """
        
        # src_outputs = [src len, batch size, enc hid dim * 2]
        # src_hidden = [batch size, dec hid dim]
        src_outputs, src_hidden = self.encoder(src_emb) 
        
        # prev_outputs = [prev_gen_text, batch size, enc hid dim * 2]
        # prev_hidden = [batch size, dec hid dim]
        prev_outputs, prev_hidden = self.encoder(prev_emb)
        
        stacked_hidden = torch.cat((src_hidden, prev_hidden), dim=1)           # [batch_size, dec hid dim * 2] 
        stacked_outputs = torch.cat((src_outputs, prev_outputs), dim=0)        # [src len + prev_gen_text, batch_size, enc hid dim * 2]
        
        return stacked_outputs, stacked_hidden