# -*- coding: utf-8 -*-
import json
import sys
import re

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR


class tensor_tfidf(nn.Module):
    
    """Tfidf information constraint - creating shorter version of input with the informative words """
    
    def __init__(self, vocab, lambda_, multi_freq=7): 
        
        super(tensor_tfidf, self).__init__()
        self.vocab = vocab
        self.lambda_ = lambda_
        self.multi_freq = multi_freq
        self.len_vocab = len(vocab)
        self.sos_token = vocab.stoi["<sos>"]
        self.eos_token = vocab.stoi["<eos>"]
        self.unk_token = vocab.stoi["<unk>"]

    def generate_ext(self, tfidf_tensor, src_ext):
        
        """
        Generating tfidf tensor for extended source text - with Out of Vocabulary words
        Input : tfidf_tensor - tfidf constraint tensor with unknown tokens
                src_ext - extended source text
        Output : tfidf_tensor_ - extended tfidf constraint tensor
        """
        
        tfidf_tensor_ = tfidf_tensor.clone()
        src_ext_ = src_ext.clone()
        
        # Getting the indexes of unknozn tokens in tfidf tensor
        unk_idx = ((tfidf_tensor_ == self.unk_token).nonzero(as_tuple=True)[0])            
        
        # Getting corresponding indexes in src_ext - we get them because their values is bigger than vocabulary
        src_idx = ((src_ext_ >= self.len_vocab).nonzero(as_tuple=True)[0])                

        #For each unk_index found, we replace by the value of the corresponding src_ext
        for i, idx in enumerate(unk_idx):
            tfidf_tensor_[idx] = src_ext_[src_idx[i]]

        return tfidf_tensor_

    def generate_tfidf(self, src, prev_tfidf, tfidf_tgt, desired_length, update=False):
        
        """
        Creating tensors composed with the words getting the best score (tfidf pondered but occurence in previous text update)
        Input : src - Current iteration text 
                prev_tfidf - previous text tfidf tensor (for penalizing updates) 
                tfidf_tgt - dictionnary of tfidf values of tokens in src
                desired_length - expected length of resulting tfidf tensor for src
                update - if update should taken into account - If True occurence from prev_tfidf are included in computation
        Output : tfidf_final- tensor of desired_length containing best scoring indexes/words 
        """
        #Calculatig pondered tfidf score
        tfidf_update = self.best_words_pondered(tfidf_tgt, src, desired_length)
        
        #If true - We need to recompute score to take into account the previous information from summary
        if update:
            prev_len = prev_tfidf.shape[0]
            update_tgt = {}
            
            #Special case for OOV words in previous text
            for idx in prev_tfidf:
                idx = idx.item()
                if idx not in tfidf_update:
                    tfidf_update[idx] = 0.
                    
            #Updating tfidf dictionary to take into account words from previous summary
            for word_idx in tfidf_update:
                tfidf_score = tfidf_update[word_idx]
                occurences = (prev_tfidf == word_idx).sum(dim=0).item()
                
                #lambda parameters to control novelty/coherence - multifreq ste to account for mean idf value
                ponder_score =  tfidf_score + self.multi_freq * self.lambda_ * (occurences / prev_len)
                update_tgt[word_idx] = ponder_score

            best_words = sorted(update_tgt, key=update_tgt.get, reverse=True)

        else:
            best_words = sorted(tfidf_update, key=tfidf_update.get, reverse=True)

        tfidf_final = self.tensor_ordering(src, best_words, desired_length)

        return tfidf_final

    def tensor_ordering(self, src, best_tfidf, desired_length):
        
        """
        Ordering tfidf tensor to preserve src words order
        Input : src - current iteration text
                best_tfidf - non ordered indexes tensor
                desired_length - desired length of tfidf final output - summary length
        Output : final - ordered index tensor
        """
        
        src = src.view(-1)
        new_ = torch.zeros_like(src)
        count = torch.count_nonzero(new_).item()
        
        # Adding values of best tfidf words to the tensor at correct position
        for word in best_tfidf:
            word_idx = torch.isin(src, word)
            word_tensor = word_idx * src
            new_ = new_ + word_tensor
            count = torch.count_nonzero(new_).item()
            if count >= desired_length:
                break
        
        # getting position of non 0 values
        nonzeros_idx = torch.nonzero(new_).view(-1)
        # keeping only non 0 values
        binary_tfidf = torch.index_select(new_, 0, nonzeros_idx)
        final = torch.cat((torch.tensor([self.sos_token]).to(binary_tfidf.device), binary_tfidf, torch.tensor([self.eos_token]).to(binary_tfidf.device)))

        return final

    def best_words_pondered(self, tfidf_dic, src_, desired_length): 

        """
        Increasing tfidf score of tokens close to highest tfidf words to favor sentence coherency - window size = 2 (multiplying score of close words by 1.5)
        Input : tfidf_dic - {token_index : tfidf value}
                src_ - current iteration text
                desired_length - desired length of tfidf final output - summary length
        Output : update - updated values dictionnary - {token_index : tfidf "coherent" value}
        """
        
        src_ = src_[1:-2]
        best_words = sorted(tfidf_dic, key=tfidf_dic.get, reverse=True)[:int(desired_length/2)]
        update = tfidf_dic

        ###Completing tfidf_dic for OOV words - we input an average tfidf score of the document for all OOVs words: 
        avg_tfidf = sum(tfidf_dic.values()) / float(len(tfidf_dic) + 0.01)   #Prevent empty tfidf matrices
        for _, word in enumerate(src_):
            word = word.item()
            if word not in tfidf_dic:
                tfidf_dic[word] = avg_tfidf
        
        #Loop to update scores for close words - window size = 2 (multiplying score of close words by 1.5)
        for i, word in enumerate(src_):
            word = word.item()
            if word in best_words:
                if i <= 1:
                    context_words = src_[i:i+2]
                if i > 1:
                    context_words = src_[i-2:i+2]
                if i == (src_.shape[0]-2):
                    context_words = src_[i-2:i]
                for words in context_words:
                    words = words.item()
                    tfidf_score_before_update = tfidf_dic[words]
                    if words == word:
                        new_score = tfidf_score_before_update * 1.2
                    else:
                        new_score = tfidf_score_before_update * 1.5
                    update[words] = new_score

        return update