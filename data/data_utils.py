import os
import pickle
import sys
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

def flatten_list(list_to_flatten):
    """transform list of list into flat list
    """
    #flat list of all items including in all lists in input
    return [item for sublist in list_to_flatten for item in sublist]

class tifdf_model():
    
    """
    Creation of the matching dictionnary between tfidf values calculated with ScikitLearn TfidfVectorizer and the Vocabulary of the dataloader.
    """
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def tokenize(self, text):
        return [str(word) for word in self.nlp(str(text))]
    
    def tfidf(self, text_list, min_freq, tfidf_file, vectorizer_train=True):
        
        """
        Generating tfidf Matrix (number of document * number of words). 
        Input : text_list - list of input batch texts
                min_freq - min frequence of occurence for words to be included in the matrix - have to be the same value as used in the vocabulary
                vectorizer_train - If True train+save new tfidf model else load existing tfidf model
                tfidf_file = path towards existing tfidf trained model.
        Output : tfidf matrix for the input document, dictionnary word to column position in matrix
        """
        
        #No options here because pre-processing made before providing input list
        vectorizer = TfidfVectorizer(tokenizer=self.tokenize, min_df=min_freq)
        
        #Loading vectorizer or creating new one
        if vectorizer_train:
            vectorizer.fit(text_list)
            
            #Saving the vectorizer for future use
            with open(tfidf_file, 'wb') as savefile:
                pickle.dump(vectorizer, savefile)
        else:
            with (open(tfidf_file, "rb")) as openfile:
                vectorizer = pickle.load(openfile)
            
        return vectorizer.transform(text_list), vectorizer.get_feature_names_out()
    
    def get_tfidfdict(self, doc, tfidf_matrix, feature_names):
                
        """
        Mapping words in batch of documents to their tfidf scores in the pretrained model
        Input : doc - batch of document
                tfidf_matrix - tfidf_matrix for the batch of document
                feature_names - dictionnary mapping words to tfidf score in the matrix
        Output dictionnary with tfidf values where keys are the word index in vocabulary and the value is the tfidf score
        """
        
        #Getting all the non zero value in the sparse matrix
        feature_index = tfidf_matrix[doc,:].nonzero()[1]
        #left part match feature names to their index in dictionnary / right part get score for the corresponding name
        tfidf_scores = zip([self.vocab.stoi[feature_names[i]] for i in feature_index], [tfidf_matrix[doc, x] for x in feature_index])
        
        return dict(tfidf_scores)