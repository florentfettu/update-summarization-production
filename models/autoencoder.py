# -*- coding: utf-8 -*-
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention
from beam_search import Possible_solutions, beam_decoder
from decoder import Decoder
from dualencoder import DualEncoder, Encoder
from torch.autograd import Variable

sys.path.append("../data")
from embeddings import create_emb_layer

class AE(nn.Module):
    def __init__(self, vocab, weights_matrix, input_dim, output_dim, enc_hid_dim, dec_hid_dim, emb_dim, enc_dropout, dec_dropout, device, use_pretrained, beam_size, min_dec_steps, num_return_seq, num_return_sum, n_gram_block):
        
        super().__init__()
        
        #######################################################
        ########### Initializing General parameters ###########
        self.vocab = vocab
        self.src_pad_idx = vocab.stoi["<pad>"]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.emb_dim = emb_dim
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout
        self.device = device
        self.use_pretrained = use_pretrained
        self.trg_len_summary = 0
        
        ####################################################
        ########### Initializing embedding layer ###########
        self.weights_matrix = weights_matrix
        
        if self.use_pretrained: 
            self.embedding = create_emb_layer(self.weights_matrix, self.vocab.stoi["<pad>"]) 
        else:
            self.embedding = nn.Embedding(self.input_dim, self.emb_dim,
                                          padding_idx=self.vocab.stoi["<pad>"])
        
        ######################################################
        ########### Initializing Encoders/Decoders ###########
        self.dualencoder = DualEncoder(self.emb_dim, self.enc_hid_dim, self.dec_hid_dim, self.enc_dropout, self.vocab)
        self.encoder = Encoder(self.emb_dim, self.enc_hid_dim, self.dec_hid_dim * 2, self.enc_dropout, self.vocab)
        self.attention = Attention(self.enc_hid_dim, self.dec_hid_dim)
        self.decoder = Decoder(self.output_dim, self.emb_dim, self.enc_hid_dim, self.dec_hid_dim, self.dec_dropout, self.attention, self.vocab)
        self.w_context = nn.Linear(self.enc_hid_dim * 2, 1, bias=False)
        self.w_hidden = nn.Linear(self.dec_hid_dim * 2, 1, bias=False)
        self.w_input = nn.Linear(self.emb_dim, 1, bias=True)
        self.w_wordtfidf = nn.Linear(self.enc_hid_dim * 2, 1, bias=False)
        self.w_hiddentfidf = nn.Linear(self.dec_hid_dim * 2, 1, bias=False)
        self.w_input_idf = nn.Linear(self.emb_dim, 1, bias=True)
        
        #################################################
        ########### Initializing beam decoder ###########
        self.beam_size = beam_size
        self.min_dec_steps = min_dec_steps
        self.num_return_seq = num_return_seq
        self.num_return_sum = num_return_sum
        self.n_gram_block = n_gram_block
        self.beam_decoder = beam_decoder(self.vocab, self.embedding, self.decoder, self.w_context, self.w_hidden, self.w_input, self.w_wordtfidf, self.w_hiddentfidf, self.w_input_idf, self.beam_size, self.min_dec_steps, self.num_return_seq, self.num_return_sum, self.n_gram_block)    

    
    def create_mask(self, src):
        """ For masking padded parts of sentences in attention layer """
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def filter_oov(self, tensor):
        """ Replace any OOV index in `tensor` with <unk> token """ 
        result = tensor.clone()
        result[tensor >= len(self.vocab)] = self.vocab.stoi["<unk>"]
        return result

    def forward(self, src, src_len, prev_gentext, tfidf_tensor, tfidf_tensor_ext, src_ext, max_oov_len, desired_length, teacher_forcing_ratio):
        
        """
        Unsupervised autoencoder for update summarization
        Input  : src - input current iteration text - [src len, batch_size]
                 src_len - input text length - [batch_size]
                 prev_gentext - generated text at previous iteration - [prev_len, batch_size]
                 tfidf_tensor - tfidf costraint tensor - [desired_length, batch_size]
                 tfidf_tensor_ext - extended tfidf tensor for OOVs - [desired_length, batch_size]
                 src_ext - extended current tensor for OOVs - [src len, batch_size]
                 max_oov_len - max number of OOVs in batch - [batch_size]
                 desired_length - desired length for tfidf constraint / summary - [batch_size]
                 teacher_forcing_ratio - default value 0.2
        Output : outputs - sequence matrix vocab * seq_len to reconstruct input text
                 output_sum_recons - sequence matrix vocab * desired_length to reconstruct input tfidf constraint
                 hidden_sum - re encoding of generated tfidf constraint
                 hidden_src - encoding of input tfidf constraint
        """
        ##################################
        ########### Embeddings ###########
        src_emb = self.embedding(src)                                                  # [src len, batch size, emb dim]
        prev_emb = self.embedding(self.filter_oov(prev_gentext))                       # [prev_gentext len, batch size, emb dim]
        tfidf_emb = self.embedding(tfidf_tensor)                                       # [desired_length, batch size, emb dim]
        
        batch_size = src.shape[1]
        trg = src # autoencoder
        trg_len = src_len.item()
        # dynamically increment summary size
        self.trg_len_summary += trg_len 
        
        ################################
        ########### Encoding ###########
        # stacked_hidden = [batch_size, dec hid dim * 2]  /  tfidf_hidden = [batch_size, dec hid dim] 
        # stacked_outputs = [src len + summary len, batch_size, enc hid dim * 2]  /  tfidf_encoder_outputs = [src len, batch_size, enc hid dim * 2]
        stacked_encoder_outputs, stacked_hidden = self.dualencoder(src_emb, prev_emb)
        tfidf_encoder_outputs, tfidf_hidden = self.encoder(tfidf_emb)
        
        ##################################################
        ############ LM - Decoding source text ###########
        input_dec = trg[0,:] 
        mask = self.create_mask(torch.cat((src, prev_gentext)))                            # [batch size, src len + trg len]
        outputs = torch.zeros(self.trg_len_summary, batch_size, self.output_dim + max_oov_len).to(self.device)
        
        for t in range(1, self.trg_len_summary): 
            # Step 1 - apply decoder
            input_dec = self.embedding(self.filter_oov(input_dec.unsqueeze(0)))
            vocab_dist, attn_dist, context_vec, stacked_hidden = self.decoder(input_dec, stacked_hidden, stacked_encoder_outputs, mask)
            
            # Step2 - Copy mechanism
            context_feat = self.w_context(context_vec)                                     # [batch_size, 1]
            decoder_feat = self.w_hidden(stacked_hidden)                                   # [batch_size, 1]
            input_feat = self.w_input(input_dec.squeeze(0))                                # [batch_size, 1]
            gen_feat = context_feat + decoder_feat + input_feat                            # [batch_size, 1] 
            p_gen = torch.sigmoid(gen_feat)                                                # [batch_size, 1]
            vocab_dist = p_gen * vocab_dist                                                # [batch_size, output_dim]
            weighted_attn_dist = (1.0 - p_gen) * attn_dist                                 # [batch_size, src len + summary len]
            extra_zeros = torch.zeros((batch_size, max_oov_len), device=vocab_dist.device) # [batch_size, OOV_len]
            extended_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1)             # [batch_size, output_dim + OOV_len]
            final_dist = extended_vocab_dist.scatter_add(dim=-1, index=torch.cat((src_ext, prev_gentext)).permute(1,0), src=weighted_attn_dist)  # [1, output_dim + OOV_len]
            
            # Step3 - Storing results
            outputs[t] = final_dist                                                        # [src_len, vocab_size]
            top1 = final_dist.argmax(1)
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            # Avoiding errors when the summary is longer than the new sequence
            if self.trg_len_summary < trg.shape[0]: 
                # if teacher forcing, use actual next token as next input. If not, use predicted token
                input_dec = trg[t] if teacher_force else top1
            else:
                input_dec = top1
        
        ######################################################################
        ############ TFIDF constraint model - Decoding tfidf text ############       
        output_sum_recons = torch.zeros(desired_length, batch_size, self.output_dim + max_oov_len).to(self.device)  # + max_oov_len
        outputs_pred = torch.zeros(desired_length, batch_size).to(self.device)
        mask_tfidf = self.create_mask(tfidf_tensor)
        input_tfidf = tfidf_tensor[0,:]
        
        for t in range(1, desired_length):
            input_tfidf = self.embedding(self.filter_oov(input_tfidf.unsqueeze(0)))

            # Step 1 - apply decoder, 
            vocab_tfidf, attn_dist_idf, context_tfidf, tfidf_hidden = self.decoder(input_tfidf, tfidf_hidden, tfidf_encoder_outputs, mask_tfidf)
            
            # Step2 - Copy mechanism
            wordtfidf_feat_idf = self.w_wordtfidf(context_tfidf)
            hiddentfidf_feat_idf = self.w_hiddentfidf(tfidf_hidden)
            input_feat_idf = self.w_input_idf(input_tfidf.squeeze(0))
            gen_idf = wordtfidf_feat_idf + hiddentfidf_feat_idf + input_feat_idf
            p_idf = torch.sigmoid(gen_idf)
            vocab_tfidf = vocab_tfidf * p_idf
            weighted_idf = (1.0 - p_idf) * attn_dist_idf 
            extra_zeros_idf = torch.zeros((batch_size, max_oov_len), device=vocab_tfidf.device)        # [batch_size, OOV_len]
            extended_vocab_idf = torch.cat([vocab_tfidf, extra_zeros_idf], dim=-1)                     # [batch_size, output_dim + OOV_len]
            final_idf = extended_vocab_idf.scatter_add(dim=-1, index=tfidf_tensor_ext.permute(1,0), src=weighted_idf)
            
            # Step3 - Storing results
            output_sum_recons[t] = final_idf
            top_idf = final_idf.argmax(1)
            outputs_pred[t] = top_idf
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            input_tfidf = tfidf_tensor[t] if teacher_force else top_idf
            
        ###########################################################################
        ############ Cosine control to ensure model respect constraint ############
        outputs_pred = F.pad(input=outputs_pred, pad=(0,0,0,src_ext.shape[0] - outputs_pred.shape[0]), mode='constant', value=self.vocab.stoi["<eos>"])
        outputs_pred = outputs_pred.type(torch.LongTensor).to(self.device)
        sum_emb = self.embedding(self.filter_oov(outputs_pred))
        _, hidden_sum = self.encoder(sum_emb)                                 # [batch_size, dec hid dim] 
        _, hidden_src = self.encoder(tfidf_emb)                               # [batch_size, dec hid dim]
        
        return outputs, output_sum_recons, hidden_sum, hidden_src
    

    def inference(self, src, src_len, prev_gentext, tfidf_tensor, tfidf_tensor_ext, desired_length, src_ext, max_oov_len, src_oovs, src_oovs_updated, update_id, beam_sum_ids, beam_summaries):
        
        """
        Generating Final summary for each text / update
        Input  : src - input current iteration text
                 src_len - input text length
                 prev_gentext - generated text at previous iteration
                 tfidf_tensor - tfidf costraint tensor
                 tfidf_tensor_ext - extended tfidf tensor for OOVs
                 desired_length - desired length for tfidf constraint / summary
                 src_ext - extended current tensor for OOV
                 max_oov_len - max number of OOVs in batch
                 src_oovs- list of OOVs words
                 src_oovs_updated - list updated of OOVs words through each timestep iterations 
                 update_id - id denoting update iteration of text (if 0 then no iteration, thus no previous generated text)
                 beam_sum_ids - list for storing final results indexes (all potential beam are reused for next iterations)
                 beam_summaries - list for storing text final results
        Output : sum_idx - final result with indexes
                 sum_sentences - final summary produced
                 beam_sum_ids - list of beam produced indexed summaries 
                 beam_summaries - list of beam produced text summaries
        """
        trg = src
        trg_len = src_len.item()
        self.trg_len_summary += trg_len
        
        ##################################
        ########### Embeddings ###########
        src_emb = self.embedding(src)                                                                      #[src len, batch size, emb dim]
        if update_id == 0:
            tfidf_tensor = tfidf_tensor[0]
            tfidf_tensor_ext = tfidf_tensor_ext[0]                                                         
            
            tfidf_emb = self.embedding(tfidf_tensor)                                                       # [desired_len, 1, emb dim]
            prev_emb = self.embedding(prev_gentext)                                                        # [summary len, 1, emb dim]
        else:
            tfidf_emb = [self.embedding(tfidf_tensor_i) for tfidf_tensor_i in tfidf_tensor]                # [desired_len, K=beam_size, emb dim]
            prev_emb = [self.embedding(self.filter_oov(prev_gentext)) for prev_gentext in prev_gentext]    # [summary len, K, emb dim]
        
        if update_id == 0:
            ########### Encoding ###########
            stacked_encoder_outputs, stacked_hidden = self.dualencoder(src_emb, prev_emb)
            tfidf_encoder_outputs, tfidf_hidden = self.encoder(tfidf_emb)
            mask = self.create_mask(torch.cat((src, prev_gentext)))                                             # [batch size, src + summary len]
            mask_tfidf = self.create_mask(tfidf_tensor)
            
            ########### Decoding ###########
            self.beam_decoder.num_return_sum = 5 #return the 5 best summaries from the 1st iteration
            sum_idx, sum_sentences = self.beam_decoder.decode(src_ext, tfidf_tensor_ext, prev_gentext, stacked_encoder_outputs, stacked_hidden, tfidf_encoder_outputs, tfidf_hidden, mask, mask_tfidf, src_oovs, src_oovs_updated, max_oov_len, desired_length)
        
        #############################
        # Case where we already generated summaries so previous iteration take into account 5 different summaries and 5 different tfidf tensors
        else:
            for i in range(len(prev_gentext)):
                
                ########### Encoding ###########
                stacked_encoder_outputs, stacked_hidden = self.dualencoder(src_emb, prev_emb[i])
                tfidf_encoder_outputs, tfidf_hidden = self.encoder(tfidf_emb[i])
                mask = self.create_mask(torch.cat((src, prev_gentext[i])))                                      # [batch size, src + summary len]
                mask_tfidf = self.create_mask(tfidf_tensor[i])
                
                ########### Decoding ###########
                self.beam_decoder.num_return_sum = 1 # return only the best summary to avoid having beam size to the power of updates
                sum_idx, sum_sentences = self.beam_decoder.decode(src_ext, tfidf_tensor_ext[i], prev_gentext[i], stacked_encoder_outputs, stacked_hidden, tfidf_encoder_outputs, tfidf_hidden, mask, mask_tfidf, src_oovs, src_oovs_updated, max_oov_len, desired_length)
                beam_sum_ids.append(sum_idx)
                beam_summaries.append(sum_sentences)
        
        
        return sum_idx, sum_sentences, beam_sum_ids, beam_summaries