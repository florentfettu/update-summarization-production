import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module): 
    
    """Attention layer - attention paid on encoder outputs for each decoding steps"""
    
    def __init__(self, enc_hid_dim, dec_hid_dim):
        
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + (dec_hid_dim * 2), dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, stacked_hidden, encoder_stacked_outputs, mask):
        """
        Input : stacked_hidden - [batch size, dec hid dim * 2]
                encoder_stacked_outputs - [src len + summary len, batch size, enc hid dim * 2]
                mask - [batch size, src + summary len]
        Output : attn_dist - [batch size, src len + summary len]
        """
        
        src_summary_len = encoder_stacked_outputs.shape[0]
        
        #repeat decoder hidden state (src len + trg len) times
        stacked_hidden = stacked_hidden.unsqueeze(1).repeat(1, src_summary_len, 1) # [batch size, src len + summary len, dec hid dim * 2]
        encoder_stacked_outputs = encoder_stacked_outputs.permute(1, 0, 2) # [batch size, src len + summary len, enc hid dim * 2]
        energy = torch.tanh(self.attn(torch.cat((stacked_hidden, encoder_stacked_outputs), dim = 2))) #[batch size, src len + summary len, dec hid dim]
    
        attention = self.v(energy).squeeze(2) #[batch size, src len + summary len]
        attention = attention.masked_fill(mask == 0, -1e10)
        attn_dist = F.softmax(attention, dim = 1) # [batch size, src len + summary len]
        
        return attn_dist