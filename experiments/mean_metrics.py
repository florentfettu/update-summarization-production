#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import spacy
import sklearn.metrics as dist
nlp = spacy.load("en_core_web_sm")


### AJOUTER ICI CALCUL DE ROUGE MOYEN POUR TOUS ET RETURN LES SCORES

def tokenize(text):
    return [tok.text.lower() for tok in nlp.tokenizer(str(text))]


def rouge_means(df):
    
    rouge_1_p = df["rouge1.precision"].mean()
    rouge_1_r = df["rouge1.recall"].mean() 
    rouge_1_f = df["rouge1.fmeasure"].mean() 
    rouge_2_p = df["rouge2.precision"].mean() 
    rouge_2_r = df["rouge2.recall"].mean() 
    rouge_2_f = df["rouge2.fmeasure"].mean() 
    rouge_L_p = df["rougeL.precision"].mean() 
    rouge_L_r = df["rougeL.recall"].mean() 
    rouge_L_f = df["rougeL.fmeasure"].mean() 
    
    return [rouge_1_p, rouge_1_r, rouge_1_f, rouge_2_p, rouge_2_r, rouge_2_f, rouge_L_p, rouge_L_r, rouge_L_f]    

def pct_reuse(src_list, update_ids, sum_list):
    
    """
    Objectif : Combien de mos sont ré-utilisés depuis le texte d'origine pour les updates et qui n'était pas présent dans le résumé de l'itération précédente
               Si lambda négatif - pct_reuse entre src et sum increase puisque on empêche l'emploi des mots de l'itération précédente
               Si lambda positif - pct_reuse entre src et sum decrease puisque on favorise l'emploi des mots de l'itération précédente qui n'existe pas forcément dans src
    """
   
    pct_reused = []
    for i in range(len(update_ids)):
        update_count = 0
        if update_ids[i] > 0:
            tok_src = tokenize(src_list[i])
            tok_summ = tokenize(sum_list[i])
            tok_summ_prev = tokenize(sum_list[i-1])
            for word in tok_summ:
                if word in tok_src:
                    if word not in tok_summ_prev:
                        update_count+= 1
                    
            #On compte le nombre de mots (normalisé) appartenant à la src
            pct_reused.append(update_count/len(tok_summ))
        
    #On retourne la moyenne des 
    return np.mean(pct_reused)

def pct_new(src_list, update_ids, sum_list):
    
    """
    Objectif : Combien de mots ccommun entre summ[i] et summ[i-1] connaissant le ration de mts commun entre src[i] et src[i-1]
               Si lambda est négatif : La mesure devrait :  nb_comm(summ[i] vs summ[i-1]) <  nb_comm(src[i] vs src[i-1])
                                       On encourage l'utilisation de mots qui n'existe pas dans la source de l'itération précédente
               Si lambda est positif : La mesure devrait :  nb_comm(summ[i] vs summ[i-1]) >  nb_comm(src[i] vs src[i-1])
                                       On encourage à réutiliser des mots qui n'existait pas avant
    """
    pct_news = []
    for i in range(len(update_ids)):
        src_count = 0
        sum_cout = 0
        
        if update_ids[i] > 0:

            tok_src_curr = tokenize(src_list[i])
            tok_src_prev = tokenize(src_list[i-1])
            tok_summ_curr = tokenize(sum_list[i])
            tok_summ_prev = tokenize(sum_list[i-1])
            for word in tok_src_curr:
                if word in tok_src_prev:
                    src_count += 1/len(tok_src_curr)
            for word in tok_summ_curr:
                
                if word in tok_summ_prev:
                    sum_cout += 1/len(tok_summ_curr)
            if src_count != 0:        
                ratio_count = sum_cout / (src_count)
            else:
                pass
            
            pct_news.append(ratio_count)
            
    return np.mean(pct_news)