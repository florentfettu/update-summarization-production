from datetime import datetime

import torch.nn as nn

def count_parameters(model):
    """Counting paramters in the autoencoder model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    """Estimating time for one epoch"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

###################
# 3 differents methods to initialize weights parameters for layers in the model
# Model has been trained by default with xavier_weights_init
def normal_weights_init(m):
    """
    Initializing model parameters to normal distribution
    Input : m - model
    """
    
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
def xavier_weights_init(m):
    """
    Initializing model parameters to uniform distribution
    Input : m - model
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param.data)
        else:
            nn.init.constant_(param.data, 0)
            
def orthogonal_weights_init(m):
    
    """
    Initializing model parameters to normal distribution such that weigths tensor are orthogonals
    Input : m - model
    """
    
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.orthogonal_(param.data)
        else:
            nn.init.constant_(param.data, 0)
            
def treat_results(results):
    
    """Homogenization of results format obtained from model"""
    treated_results = []
    results = results.generated_summaries.to_list()
    for summaries in results:
        temp = []
        if type(summaries[0])  == list:
            for i in range(len(summaries)):
                temp.append(summaries[i][0])
            treated_results.append(temp)
        else:
            treated_results.append(summaries)
            
    return treated_results