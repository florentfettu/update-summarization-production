import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    
    """Decoder layer - Generating probability distribution over vocabulary"""
    
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, vocab, use_pretrained=True):
        
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim , dec_hid_dim * 2)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, stacked_hidden, encoder_stacked_outputs, mask):
        
        """
        Input : input - input index words - [batch size]
                stacked_hidden - [batch size, dec hid dim * 2]
                encoder_stacked_outputs - [src len + prev_gen_txt, batch size, enc hid dim * 2]
                mask - [batch size, src len + prev_gen_txt]
        Output : vocab_dist - probability distrubtion over vocabulary
                 attn_dist - attention values
                 context_vec - context vector of attention over input
                 hidden - text representation after rnn
        """
        
        
        embedded = self.dropout(input)                                               # [1, batch size, emb dim]
        attn_dist = self.attention(stacked_hidden, encoder_stacked_outputs, mask)    # [batch size, src len + prev_gen_txt]
        attn_dist = attn_dist.unsqueeze(1)                                           # [batch size, 1, src len + prev_gen_txt]
        encoder_stacked_outputs = encoder_stacked_outputs.permute(1, 0, 2)           # [batch size, src len + prev_gen_txt, enc hid dim * 2]
        
        #inputing attention to input sequence
        context_vec = torch.bmm(attn_dist, encoder_stacked_outputs)                  # [batch size, 1, enc hid dim * 2]
        context_vec = context_vec.permute(1, 0, 2)                                   # [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, context_vec), dim = 2)                      # [1, batch size, (enc hid dim * 2) + emb dim]
        # output =[seq len, batch size, dec hid dim * 2 * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim * 2]
        output, hidden = self.rnn(rnn_input, stacked_hidden.unsqueeze(0))
        
        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim * 2]
        # hidden = [1, batch size, dec hid dim * 2]
        # this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)                                                # [batch size, emb dim]
        output = output.squeeze(0)                                                    # [batch size, dec hid dim * 2]
        context_vec = context_vec.squeeze(0)                                          # [batch size, enc hid dim * 2]
        
        #Output probability distribution over vocabulary
        prediction = self.fc_out(torch.cat((output, context_vec, embedded), dim = 1)) # [batch size, output dim]
        vocab_dist = F.softmax(prediction, dim=-1)                                    # [batch size, output dim]
        
        return vocab_dist, attn_dist.squeeze(1), context_vec, hidden.squeeze(0)