import os
import pickle
import sys
#from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import spacy
import torch
import torch.nn as nn
from preprocess import TextPreprocessing
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

sys.path.append("..")
from data_utils import *
from vocab import *

class TrecDataset(Dataset, tifdf_model):
    
    """
    Transforming and formatting the dataset in order to be batched
    """
    
    def __init__(self, file, vocabulary, min_freq, vocab_size, remove_stopwords, remove_punctuation, tfidf_file, vectorizer_train=True):
       
        # load the data
        self.df = pd.read_csv(file)
        
        # Index list for each new document/event
        self.batch_indices = self.df.index[self.df['update_id'] == 0].tolist() 
        self.nlp = spacy.load("en_core_web_sm")
        
        # Get update_id, update_text and nugget_text
        self.update_id = self.df["update_id"] 
        self.update_text = self.df["update_text"]
        self.nugget_text = self.df["nugget_text"]
        
        # Data cleaning + removing stopwords and punctuation if needed - set both values to True to reproduce paper results
        self.preprocess = TextPreprocessing(remove_stopwords, remove_punctuation)
        self.update_text = self.update_text.apply(self.preprocess.regex_pattern_removal)
        
        #Getting tfidf matrix and dictionnary of walues per word
        if vectorizer_train:
            self.update_tfidf, self.feature_names = tifdf_model.tfidf(self, self.update_text, min_freq, tfidf_file=tfidf_file, vectorizer_train=True)
            
        else: 
            try:
                self.update_tfidf, self.feature_names = tifdf_model.tfidf(self, self.update_text, min_freq, tfidf_file=tfidf_file, vectorizer_train=False)
            except FileNotFoundError:
                print('You have to provide a vectorizer file - try vectorizer_train=True the first time you train the model')
            
        
        #Building/Loading vocabulary
        if not vocabulary:
            self.vocab = Vocabulary(min_freq, vocab_size)
            self.vocab.build_vocabulary(self.update_text.to_list())
            
        else:
            #reuse vocab from training data for valid/test data
            self.vocab = vocabulary 
        
    def __len__(self):
        # return the length of the batch_indices - 1 to avoid index out of range
        return len(self.batch_indices) - 1 
    
    def __getitem__(self, index):
        
        """
        Formatting the information in an individual item/batch
        """
        
        start_idx = self.batch_indices[index]                                              # Starting position/index of document/event (can include multiple index = update per document)
        end_idx = self.batch_indices[index+1]                                              # Ending position of document
        
        #Getting update ids, update texts, reference/nugget summary, and tfidf values associated to each document
        update_id = self.update_id[start_idx:end_idx]
        update_text = self.update_text[start_idx:end_idx]
        nugget_text = self.nugget_text[start_idx:end_idx]
        update_tfidf = self.update_tfidf[start_idx:end_idx,:]
        
        numericalized_text_batch = []
        src_ids_ext_batch = []
        oovs = []
        max_oov_len = []
        tfidf_batch = []
        
        
        for i, text in enumerate(update_text):
            #adding special tokens and numericalizing with vocabulary index the texts
            numericalized_text = [self.vocab.stoi["<sos>"]]
            numericalized_text += self.vocab.numericalize(text)                              # add the numericalized text
            numericalized_text.append(self.vocab.stoi["<eos>"])                              # append the end of sentence token
            numericalized_text_batch.append(torch.tensor(numericalized_text))
            
            #Save the tfidf tensor in batch info
            tfidf = tifdf_model.get_tfidfdict(self, i, update_tfidf, self.feature_names)
            tfidf_batch.append(tfidf)
            
            # Extend vocabulary and numericalized for OOVs words / for pointer-generator
            src_ids_ext = [self.vocab.stoi["<sos>"]]
            src_ids_ext += self.vocab.oov_ids_extended_vocab(text)[0]                        # Store the version of the encoder batch that uses article OOV ids
            src_ids_ext.append(self.vocab.stoi["<eos>"])                                     # append the end of sentence token
            src_ids_ext_batch.append(torch.tensor(src_ids_ext))
            oovs.append(self.vocab.oov_ids_extended_vocab(text)[1])                          # Store source text OOVs themselves
        
        # retrieve the max oov len per batch
        max_oov_len.append(max([len(oov) for oov in oovs]))
            
        return list(update_id), numericalized_text_batch, src_ids_ext_batch, oovs, max_oov_len, list(nugget_text), tfidf_batch


class MyCollate(): 
    
    """
    Calling items per batch and collating individual information to create/form the batch
    """
    
    def __init__(self, pad_idx, vocabulary=None):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        update_id = [item[0] for item in batch]                                                # Extract update_id from batch
        update_id = flatten_list(update_id)                                                    # flatten list of list
        
        update_text = [item[1] for item in batch]                                              # Extract update_text from batch
        update_text = flatten_list(update_text)                                                # flatten list of list
        
        lengths = torch.tensor([len(sequence) for sequence in update_text])                    # store the length of each sequence
        
        update_text = pad_sequence(update_text, batch_first=False, padding_value=self.pad_idx) # pad according to the max sequence length within each batch to avoid unnecessary padding
        
        #Generating text with OOVs indexes and not Unknown tokens
        src_ids_ext = [item[2] for item in batch]
        src_ids_ext = flatten_list(src_ids_ext)                                                # flatten list of list
        src_ids_ext = pad_sequence(src_ids_ext, batch_first=False, padding_value=self.pad_idx) 
        
        #Storing OOVs words indexes in batch info
        oovs = [item[3] for item in batch]
        oovs = flatten_list(oovs)
        
        #Storing max length of OOVs for the batch
        max_oov_len = [item[4] for item in batch]
        max_oov_len = flatten_list(max_oov_len)
        
        #Storing reference summary for the batch
        nugget_text = [item[5] for item in batch]
        nugget_text = [flatten_list(nugget_text)]
        
        #Storing tfidf dictionnary for each index words in the batch
        tfidf_dict = [item[6] for item in batch]
        tfidf_dict = flatten_list(tfidf_dict)
        
        return torch.tensor(update_id), update_text, lengths, src_ids_ext, oovs, max_oov_len[0], nugget_text, tfidf_dict
    

def get_loader(file, min_freq, vocab_size, remove_stopwords, remove_punctuation, tfidf_file, vectorizer_train=False, vocabulary=None, batch_size=1, shuffle=False):
    
    """
    Loading dataset and creating batch of data
    Input : file - path to the dataset to be batch
            min_freq - min number of occurence for a word to be included in the vocabulary
            vocab_size - Max number of words included in vocabulary
            remove_stopwords - preprocessing text to remove stopwords from text before indexing and batching
            remove_punctuation - preprocessing text to remove punctuation from text before indexing and batching
            tfidf_file - path of pretrained tfidf model
            vectorizer_train - True/False - train tfidf model
            vocabulary - Provide vocabulary - When None create a new vocabulary for model.
            batch_size - Number of documet by batch - if 1 then 1 documents (with T updates) per batch
            shuffle - Not shuffling data in batch - we need to preserve update order
    """
    
    #Formatting dataset
    dataset = TrecDataset(file, vocabulary, min_freq, vocab_size, remove_stopwords, remove_punctuation,  tfidf_file, vectorizer_train)
    
    #Getting vocab
    vocab = dataset.vocab
    pad_idx = vocab.stoi["<pad>"]
    
    #Transforming dataset into batchs
    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        collate_fn=MyCollate(pad_idx=pad_idx))
    
    #If vocabulary not trained, not returning it.
    if vocabulary:
        return loader
    else:
        return loader, vocab