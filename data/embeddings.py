from collections import Counter

import numpy as np
import spacy
import torch
import torch.nn as nn
import gcsfs
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def loadGloveModel(File):
    
    """ glove model files are in a word - vector format. One can use this small snippet of code to load a pretrained glove file 
    Input : File - Local glove file store in embeddings folder
    Output : gloveModel - Dictionnary matching Word to their embedding vectors
    """
    logger.info("Loading Glove Model")
    f = open(File,'r')
        
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    
    logger.info(f"{len(gloveModel)} words loaded!")
    
    return gloveModel


def weights_matrix_embedding(glove_model, vocab, vocab_size, emb_dim):
    
    """
    Loading embedding matrix for Vocabulary
    Input : glove_model - dictionnary of loaded embedding model with all words with their matching embedding
            vocab - dictionnary continaing dataset vocabulary and indexes
            vocab_size : nb of words stored in vocab
            emb_dim : dimension of embeddings - must correspond to embedding model
    Output : weights_matrix - matrix of indexes of vocabulary with their associated embedding
    """
    
    weights_matrix = torch.zeros((vocab_size, emb_dim))
    words_found = 0

    for i, word in enumerate(vocab):                                                            # use values itos dict
        try: 
            weights_matrix[i] = torch.from_numpy(glove_model[word])
            words_found += 1
        except KeyError:
            weights_matrix[i] = torch.randn(emb_dim)                                            # Initialize a random tensor following a normal distribution for the oovs
    
    logger.info(f"{words_found} words over {vocab_size} were found in glove embeddings")
    
    return weights_matrix


def create_emb_layer(weights_matrix, pad_idx, non_trainable=False):
    
    """
    Loading weigth matrix into embeddings tensor
    Input : weights_matrix - Embedding weigth matrix for vocabulary
            pad_idx - Padding index to create special embedding in pytorch embeddings
            non_trainable - gradient backpropagation on embedding tensors
    Output : emb_layer - embedding tensor
    """
    
    # Initialize embeddings 
    vocab_size, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
    
    # Using pre-trained embeddings
    emb_layer.load_state_dict({'weight': weights_matrix})
    
    # embedding layer has one trainable parameter called weight, which is, by default, set to be trained.
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer