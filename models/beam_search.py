import re
import torch
import torch.nn as nn
import torch.nn.functional as F

class Possible_solutions():
    
    """Store beam solutions during process"""
    
    def __init__(self, tokens, log_probs, stacked_hidden, tokens_idf, log_probs_idf, tfidf_hidden):#, coverage):
        
        """
        Input : tokens - List of tokens associated to original text reconstruction
                log_probs - List of tokens log probabilities
                tokens_idf - List of tokens associated tfidf tensor reconstruction
                log_probs_idf - List of tokens_idf log probabilities
                stacked_hidden - Reconstruction model hidden layer
                tfidf_hidden - tfidf constraint model hidden layer
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.tokens_idf = tokens_idf
        self.log_probs_idf = log_probs_idf
        self.stacked_hidden = stacked_hidden
        self.tfidf_hidden = tfidf_hidden

    def extend(self, token, log_prob, stacked_hidden, token_idf, log_idf, tfidf_hidden):
        
        """
        Extend list of beam solutions with the best predicted tokens
        Input : token - best predicted token
                log_prob - log probability associated with the tokens
                stacked_hidden - preserving hideen encoder representations in all decoding steps
                token_idf - best predicted tfidf token
                log_idf - log probability associated with the tfidf tokens
                tfidf_hidden - preserving hideen tfdif encoder representations in all decoding steps
        Output : beam_size² - list with all possible solution
        """
        return Possible_solutions(tokens=self.tokens + [token],     
                          log_probs=self.log_probs + [log_prob],
                          stacked_hidden=stacked_hidden, 
                          tokens_idf=self.tokens_idf + [token_idf],     
                          log_probs_idf=self.log_probs_idf + [log_idf],
                          tfidf_hidden=tfidf_hidden)
    
    def n_gram_blocking(self, n):
        """ n-gramm blocking function preventing repeating identical sequence in output as in Paulus, et al. (2017) """
        return self.tokens[-n:]
    
    @property
    def latest_token(self):
        """ return last token for input of decoder """
        return self.tokens[-1]
    
    @property
    def latest_token_idf(self):
        """ return last tfidf token for input of decoder """
        return self.tokens_idf[-1]
    
    @property
    def avg_log_prob(self):
        """Estimating sequence average log probability to select best sequence at each beam step """
        return sum(self.log_probs) / len(self.tokens)

    
class beam_decoder(nn.Module):
    
    """Beam decoder - generate k potential solutions at each steps and selecting best ones to decode sequence"""
    
    def __init__(self, vocab, embeddings, decoder, context_linear, hidden_linear, input_linear, context_idf , hiddentfidf, input_idf, beam_size, min_dec_steps, num_return_seq, num_return_sum, n_gram_block):
        
        """
        Input : vocab - trained data vocabulary 
                embeddings - embedding layers of the model
                decoder - decoding part of the model
                context_linear / hidden_linear / input_linear - linear layers for copy mechanism for reconstruction text
                context_idf / hiddentfidf / input_idf - linear layers for copy mechanism for tfidf constraints text
                beam_size - size of the beam search (number of possible solutions at each steps)
                min_dec_steps - Minimum sentence length that we have to produce
                num_return_seq - Minimum of summaries beam search should return
                num_return_sum - Final number of summaries returned
                n_gram_block - size of blocking window
        """
        super(beam_decoder, self).__init__()
        
        # Beam Search parameters
        self.beam_size = beam_size
        self.min_dec_steps = min_dec_steps
        self.num_return_seq = num_return_seq
        self.num_return_sum = num_return_sum
        self.output_dim = len(vocab)
        self.n_gram_block = n_gram_block
        
        #Model pretrained elements
        self.vocab = vocab
        self.embedding = embeddings
        self.decoder = decoder
        self.w_context = context_linear
        self.w_hidden = hidden_linear
        self.w_input = input_linear
        self.w_context_idf = context_idf
        self.w_hiddentfidf = hiddentfidf
        self.w_input_idf = input_idf
        
    def sort_hyps(self, hyps):
        """Sort hypotheses according to their log probability."""
        return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
    
    def filter_unk(self, idx):
        """filter unknown tokkens in input for embedding initialization"""
        return idx if idx < len(self.vocab) else self.vocab.stoi["<unk>"]


    def decode(self, src_ext, tfidf_tensor_ext, summary, stacked_encoder_outputs, stacked_hidden, tfidf_encoder_outputs, tfidf_hidden, mask, mask_tfidf, src_oovs, src_oovs_updated, max_oov_len, desired_length): 
        
        """
        Beam Search decoding
        Input : src_ext - source to summarize - [src len, batch size]
                tfidf_tensor_ext - tfidf constraint tensor - [sum_len, batch size]
                summary - previously generated summary - [sum_len, batch size]
                stacked_encoder_outputs - encoding of src + summary - [src + sum_len, batch size, enc hid dim * 2]
                stacked_hidden - encoded representation of input text - [batch size, dec hid dim * 2]
                tfidf_encoder_outputs - encoded representation of tfidf constraint model - [sum_len, batch size, enc hid dim * 2]
                tfidf_hidden - encoded representation of tfidf constraint - [batch size, dec hid dim * 2]
                mask - mask tensor - [batch size, src + sum_len]
                mask_tfidf - mask tensor for tfidf - [batch size, sum_len]
                src_oovs - list of out of vocabulary words
                src_oovs_updated - list of out of vocabulary words with previous information retained by model
                max_oov_len - max number of OOVs per batch
                desired_length - expected summary length
        Output : hyp_idx - list of summaries in idx for inclusion into next iteration timestep
                 hyp_results - list of string summary for final output/display of results
        """
        #Storing the k best hypothesis out of beam search - serves as final summaries
        best_hyps_all = [] 
        
        # Creating hypotheses
        hyps = [Possible_solutions(tokens=[self.vocab.stoi["<sos>"]], log_probs=[0.0],
                                   stacked_hidden=stacked_hidden, tokens_idf=[self.vocab.stoi["<sos>"]], 
                                   log_probs_idf=[0.0], tfidf_hidden=tfidf_hidden)]
        
        # Storing result for specific idx sentence
        sequence_results = []
        bin_final = []
        
        # K = number of running hypotheses  
        # Decoding sentence
        for t in range(desired_length):
            num_orig_hyps = len(hyps)
            input_dec = [self.filter_unk(hyp.latest_token) for hyp in hyps]
            input_dec = torch.tensor(input_dec, dtype=torch.long, device=src_ext.device)
            input_dec = self.embedding(input_dec)                                               # [K, emb_dim] 
            
            input_fin = [self.filter_unk(hyp.latest_token_idf) for hyp in hyps]
            input_fin = torch.tensor(input_fin, dtype=torch.long, device=src_ext.device)
            input_fin = self.embedding(input_fin) 

            stacked_hidden_hyp = torch.cat([hyp.stacked_hidden for hyp in hyps], dim=0)         # [K, dec hid dim * 2]
            tfidf_hidden_hyp = torch.cat([hyp.tfidf_hidden for hyp in hyps], dim=0)
            
            encoder_outputs_hyp = torch.cat([stacked_encoder_outputs for _ in hyps], dim=1)     # [src + summary len, K, enc_dim]
            tfidf_encoder_outputs_hyp = torch.cat([tfidf_encoder_outputs for _ in hyps], dim=1)
            
            src_mask_hyp = torch.cat([mask for _ in hyps], dim=0)                               # [K, src + summary len]
            mask_tfidf_hyp = torch.cat([mask_tfidf for _ in hyps], dim=0)
            
            ########## Decoder bloc   #########
            # Step 1 - apply decoder
            vocab_dist, attn_dist, context_vector, stacked_hidden_hyp = self.decoder(input_dec.unsqueeze(0), stacked_hidden_hyp, encoder_outputs_hyp, src_mask_hyp)
            vocab_tfidf, attn_tfidf, context_tfidf, tfidf_hidden_hyp = self.decoder(input_fin.unsqueeze(0), tfidf_hidden_hyp, tfidf_encoder_outputs_hyp, mask_tfidf_hyp)
            
            # Step 2 - apply linear transfromation for copy mechanism
            context_feat = self.w_context(context_vector)
            decoder_feat = self.w_hidden(stacked_hidden_hyp)
            input_feat = self.w_input(input_dec)
            wordtfidf_feat_idf = self.w_context_idf(context_tfidf)
            hiddentfidf_feat_idf = self.w_hiddentfidf(tfidf_hidden_hyp)
            input_feat_idf = self.w_input_idf(input_fin.squeeze(0))
             
            # Step 3 - concat feature vectors
            gen_feat = context_feat + decoder_feat + input_feat
            gen_idf = wordtfidf_feat_idf + hiddentfidf_feat_idf + input_feat_idf
            
            # Step 4 - calculate p_gen and new weighted dist and output with p_gen
            p_gen = torch.sigmoid(gen_feat)
            p_idf = torch.sigmoid(gen_idf)
            
            # Step 5 - compute prob distribution over extended vocabulary
            vocab_dist = p_gen * vocab_dist
            vocab_tfidf = vocab_tfidf * p_idf
            weighted_attn_dist = (1.0 - p_gen) * attn_dist
            weighted_idf = (1.0 - p_idf) * attn_tfidf
            
            # Step 6 - manage OOV words with extra_zeros
            extra_zeros = torch.zeros((num_orig_hyps, max_oov_len), device=vocab_dist.device) 
            extended_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1)
            final_dist = extended_vocab_dist.scatter_add(dim=-1, index=torch.cat((src_ext, summary)).permute(1,0), src=weighted_attn_dist)   #[K, vocab_dist]
            extra_zeros_idf = torch.zeros((num_orig_hyps, max_oov_len), device=vocab_tfidf.device)
            extended_vocab_idf = torch.cat([vocab_tfidf, extra_zeros_idf], dim=-1)
            final_idf = extended_vocab_idf.scatter_add(dim=-1, index=tfidf_tensor_ext.permute(1,0), src=weighted_idf)                        #[K, vocab_dist]
            
            #For each hypothesis we need to take max value between LM model and TFIDF model
            final_mat = torch.zeros_like(final_idf).to(final_idf.device)
            for idx in range(final_idf.shape[0]):
                final_dist_i = final_dist[idx,:].unsqueeze(0)
                final_idf_i = final_idf[idx,:].unsqueeze(0)
                final_mat_i = torch.amax(torch.cat((final_idf_i, final_dist_i), dim=0), dim=0).unsqueeze(0)
                final_mat[idx] = final_mat_i                                                                                                #[K, vocab_dist]
            
            #Taking the max probability here
            log_probs = torch.log(final_mat)
            log_probs_idf = torch.log(final_mat)
            
            #Taking the best options through log distribution
            topk_probs, topk_ids = torch.topk(log_probs, k=self.beam_size * 2, dim=-1)                           #[K, beam_size*2]
            topk_probs_idf, topk_ids_idf = torch.topk(log_probs_idf, k=self.beam_size * 2, dim=-1)               #[K, beam_size*2]
            
            #Beam decoding part  
            all_hyps = []
            for i in range(num_orig_hyps):
                h_i = hyps[i]
                hidden_state_i = stacked_hidden_hyp[i].unsqueeze(0)                                              # [1, dec hid dim * 2]
                hidden_idf_i = tfidf_hidden_hyp[i].unsqueeze(0)                                                  # [1, dec hid dim * 2]
                
                for j in range(self.beam_size * 2):
                    if topk_ids[i, j].item() == self.vocab.stoi["<unk>"] or topk_ids_idf[i, j].item() == self.vocab.stoi["<unk>"]:
                        pass                   
                    else:
                        if t > 0:
                            if topk_ids[i, j].item() in h_i.n_gram_blocking(self.n_gram_block) or topk_ids_idf[i, j].item() in h_i.n_gram_blocking(self.n_gram_block):
                                pass 
                            else:
                                new_hyp = h_i.extend(token=topk_ids[i, j].item(), log_prob=topk_probs[i, j].item(), stacked_hidden=hidden_state_i,
                                                    token_idf=topk_ids_idf[i, j].item(), log_idf=topk_probs_idf[i, j].item(), tfidf_hidden=hidden_idf_i)
                        else:
                            new_hyp = h_i.extend(token=topk_ids[i, j].item(), log_prob=topk_probs[i, j].item(), stacked_hidden=hidden_state_i,
                                                token_idf=topk_ids_idf[i, j].item(), log_idf=topk_probs_idf[i, j].item(), tfidf_hidden=hidden_idf_i)
                    all_hyps.append(new_hyp)
            
            # hyps include 5 besthypotheses for each timestep
            hyps = [] 
            for hyp in self.sort_hyps(all_hyps):
                #Including hypothesis to results if model generate eos and is longest than min_dec_steps
                if hyp.latest_token == self.vocab.stoi["<eos>"]:
                    if t >= self.min_dec_steps:
                        sequence_results.append(hyp)
                
                #Else passing hypothesis to pursue beam decoding
                else:
                    hyps.append(hyp)
                if len(hyps) == self.beam_size or len(sequence_results) == self.beam_size:
                    break
            if len(sequence_results) == self.beam_size:
                break
            
        # Reached max decode steps but not enough results
        if len(sequence_results) < self.num_return_seq:
            sequence_results = sequence_results + hyps[:self.num_return_seq - len(sequence_results)]
        
        #Sorting results by log probability 
        sorted_results = self.sort_hyps(sequence_results)
        best_hyps = sorted_results[:self.num_return_seq]
        
        #best_hyps_all include 5 best hypotheses of last timestep
        best_hyps_all.extend(best_hyps) 
        sorted_all_hyps = self.sort_hyps(best_hyps_all)
        best_summary = sorted_all_hyps[:self.num_return_sum]
        # we can't convert list to tensors when elements have different sizes 
        token_summary = [hyp.tokens for hyp in best_summary]        
        max_length = max([len(element) for element in token_summary])
        padded = [row + [2] * (max_length - len(row)) for row in token_summary] # we pad with <eos> token
        hyp_idx = torch.tensor([i for i in padded])
        
        #transforming token list into words for final summary 
        hyp_words = [self.vocab.invert_numericalize_extended(src_oovs, src_oovs_updated, hyp.tokens) for hyp in best_summary]
        hyp_results = [postprocess(words, skip_special_tokens=True, clean_up_tokenization_spaces=True) for words in hyp_words]
        
        return hyp_idx, hyp_results
    
def postprocess(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True):
    
    """
    Cleaning and transforming list of tokens into full sentence
    Input : tokens - token list
            skip_special_tokens - filter some special tokens
            clean_up_tokenization_spaces - managing special case of english
    Output : out_string - text string
    """
    if skip_special_tokens:
        tokens = [t for t in tokens if not is_special(t)]

    out_string = ' '.join(tokens)

    if clean_up_tokenization_spaces:
        out_string = clean_up_tokenization(out_string)

    return out_string

def is_special(token):
    res = re.search("\<[a-z0-9]+\>", token)
    if res is None:
        return False
    return token == res.group()

def clean_up_tokenization(out_string):
    """
    Reference : transformers.tokenization_utils_base
    Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
    Args:
        out_string (:obj:`str`): The text to clean up.
    Returns:
        :obj:`str`: The cleaned-up string.
    """
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string