import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

def loss_estimation(output, trg, prev_text, vocab, update_id, idx, lambda_, epoch, include_prev_epoch, output_idf, hidden_sum, hidden_src, trg_idf):
    
    """
    Estimation of general model loss : reconstruction of text + tfidf constraint reconstruction + cosine loss for regularization
    Input : output - Generated text output by the model 
            trg - target text / original text since self-supervised
            prev_text - previously generated text
            vocab - trained vocabulary
            update_id - News iteration in the document batch
            idx - current text index in batch
            lambda_ - Novelty / Coherence control parameter
            epoch - current training epoch
            include_prev_epoch - When to include previous text in reconstruction loss
            output_idf - Generated tfidf constraint tensorby the model
            hidden_sum - re-encoded representation of tfidf constraint model 
            hidden_src - encoded representation of text
            trg_idf - target tfidf constraint model
    """
    
    #Defining Error losses
    criterion = nn.NLLLoss(ignore_index=vocab.stoi["<pad>"]) 
    criterion_idf = nn.NLLLoss(ignore_index=vocab.stoi["<pad>"])
    criterion_sum = nn.CosineSimilarity(dim=1)
    
    #pad output to have same length as target - add tensors n of [1, vocab_size] with 0 values
    output_pad = F.pad(input=output, pad=(0,0,0, trg.shape[0] - output.shape[0]), mode='constant', value=0)
    #Replace 0 values in the new tensors by 1 at pad position.
    output_pad[output.shape[0]:, vocab.stoi["<pad>"]] = 1  
    
    # Reconstruction Loss depending on updates
    if update_id[idx].item() == 0:
        rec_loss = criterion(output_pad, trg) 
        
    #If it's not the first iteration of the document
    else:          
        prev_text_pad = F.pad(input=prev_text.squeeze(1), pad=(0, trg.shape[0] - prev_text.shape[0]), mode='constant', value=vocab.stoi["<eos>"]) 

        #If we include prev_text we include the current text reconstruction loss
        if epoch > include_prev_epoch:
            rec_loss = criterion(output_pad, trg) + lambda_ * criterion(output_pad, prev_text_pad)

        #If we do not include prev_text we only consider text reconstruction loss
        else:
            rec_loss = criterion(output_pad, trg)
    
    # tfdif Loss
    output_idf_pad = F.pad(input=output_idf, pad=(0,0,0, trg_idf.shape[0] - output_idf.shape[0]), mode='constant', value=0) 
    output_idf_pad[output_idf_pad.shape[0]:, vocab.stoi["<pad>"]] = 1
    idf_loss = criterion_idf(output_idf_pad, trg_idf)
    
    # Esunring that prediction enbales to reconstruct target too
    sum_recons = -1 * (criterion_sum(hidden_sum, hidden_src) + 1) / 2
    
    #Final loss
    loss = rec_loss + idf_loss + sum_recons
    
    return loss, rec_loss, idf_loss, sum_recons