# -*- coding: utf-8 -*-
import pickle
import random
import re
import sys
import time
import math
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.append("data")
from dataloader import *
from embeddings import *

sys.path.append("models")
from autoencoder import *

sys.path.append("experiments")
from run_utils import *
from procedures import Procedure
from mean_metrics import *

sys.path.append("configs")
import config

import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Update summarization using unsupervised autoencodder")

################ data ################
# ENELVER GLOVE PATH AT THE END - GS
parser.add_argument("--glove", default=config.data["glove"], help="glove embeddings file path")
parser.add_argument("--train", default=config.data["train"], help="data path for training")
parser.add_argument("--valid", default=config.data["valid"], help="data path for validation")
parser.add_argument("--test", default=config.data["test"], help="data path for testing")
parser.add_argument("--min_freq", default=config.data["min_freq"], type=int, help="minimum frequency needed to include a token in the vocabulary")
parser.add_argument("--vocab_size", default=config.data["vocab_size"], type=int, help="maximum size of the vocabulary")
parser.add_argument("--remove_stopwords", default=config.data["remove_stopwords"], type=bool, help="remove stopwords within preprocessing steps")
parser.add_argument("--remove_punctuation", default=config.data["remove_punctuation"], type=bool, help="remove punctuation within preprocessing steps")

################ Summarization parameters ################
parser.add_argument("--length_ratio_train", default=config.sum_param["length_ratio_train"], type=float, help="Comrpession ratio when training")
parser.add_argument("--length_ratio_test", default=config.sum_param["length_ratio_test"], type=float, help="Comrpession ratio when generating final")
parser.add_argument("--lambda_", default=config.sum_param["lambda_"], type=float, help="Update summary's parameters (positive for coherence and negative value for novelty)")

################ training parameters ################
parser.add_argument("--save_tfidf_path", default=config.trainer["save_tfidf_path"], help="data for pretrained TFIDF Model")
parser.add_argument("--save_model_path", default=config.trainer["save_model_path"], help="data for pretrained TFIDF Model")
parser.add_argument("--save_vocab_path", default=config.trainer["save_vocab_path"], help="data for pretrained TFIDF Model")
parser.add_argument("--min_seq_len", default=config.trainer["min_seq_len"], type=int, help="Minimum length of sentence for training")
parser.add_argument("--max_seq_len", default=config.trainer["max_seq_len"], type=int, help="Maximum length of sentence for training")
parser.add_argument("--max_numel_per_batch", default=config.trainer["max_numel_per_batch"], type=int, help="Number of updates by documents considered")
parser.add_argument("--accumulation_steps", default=config.trainer["accumulation_steps"], type=int, help="mini batch size when training")
parser.add_argument("--vocab_save", default=config.trainer["vocab_save"], type=bool, help="generate summary results with rouge scores")
parser.add_argument("--model_save", default=config.trainer["model_save"], type=bool, help="generate summary results with rouge scores")
parser.add_argument("--train_tfidf", default=config.trainer["train_tfidf"], type=bool, help="Training TFIDF model on training dataset")
parser.add_argument("--model_file", default=config.trainer["model_file"], help="relative path locating the model saved")
parser.add_argument("--vocab_file", default=config.trainer["vocab_file"], help="relative path locating the vocabulary saved")
parser.add_argument("--tfidf_file", default=config.trainer["tfidf_file"], help="relative path locating the tfidf model saved")

################ model parameters ################
parser.add_argument("--emb_dim", default=config.model_parameters["emb_dim"], type=int, help="dimension number of glove pre-trained word vectors")
parser.add_argument("--enc_hid_dim", default=config.model_parameters["enc_hid_dim"], type=int, help="hidden state dimension of the encoder")
parser.add_argument("--dec_hid_dim", default=config.model_parameters["dec_hid_dim"], type=int, help="hidden state dimension of the decoder")
parser.add_argument("--enc_dropout", default=config.model_parameters["enc_dropout"], type=float, help="probability a neuron being deactivated in the encoder")
parser.add_argument("--dec_dropout", default=config.model_parameters["dec_dropout"], type=float, help="probability a neuron being deactivated in the decoder")
parser.add_argument("--use_pretrained", default=config.model_parameters["use_pretrained"], type=bool, help="use glove pre-trained word vectors")
parser.add_argument("--fxed_len_updates", default=config.model_parameters["fxed_len_updates"], type=bool, help="use fixed summary length or incrementally increase summary size with updates")
parser.add_argument("--weights_init", default=config.model_parameters["weights_init"], help="weight initialization: xavier/orthogonal/normal")

################ model hyperparameters ################
parser.add_argument("--optimizer", default=config.model_hyperparameters["optimizer"], help="optimization algorithm that will update model's parameters")
parser.add_argument("--learning_rate", default=config.model_hyperparameters["learning_rate"], type=float, help="optimization algorithm that will update model's parameters")
parser.add_argument("--weight_decay", default=config.model_hyperparameters["weight_decay"], type=float, help="optimization algorithm that will update model's parameters")
parser.add_argument("--clip", default=config.model_hyperparameters["clip"], type=int, help="optimization algorithm that will update model's parameters")
parser.add_argument("--include_prev", default=config.model_hyperparameters["include_prev"], type=bool, help="optimization algorithm that will update model's parameters")
parser.add_argument("--include_prev_epoch", default=config.model_hyperparameters["include_prev_epoch"], type=int, help="optimization algorithm that will update model's parameters")

################ beam ################
parser.add_argument("--size", default=config.beam["size"], type=int, help="Beam size")
parser.add_argument("--min_steps", default=config.beam["min_steps"], type=int, help="Minimum of words to be generated")
parser.add_argument("--num_return_seq", default=config.beam["num_return_seq"], type=int, help="Number of sequence returned by beam")
parser.add_argument("--num_return_sum", default=config.beam["num_return_sum"], type=int, help="Number of summaries generated by the model for each text")
parser.add_argument("--n_gram_block", default=config.beam["n_gram_block"], type=int, help="Windows size for non repetitive words")

################ Runs experiment ################
parser.add_argument("--writer", default=config.runs["writer"], type=bool, help="Tracking loss on Tensorboard")
parser.add_argument("--run_name", default=config.runs["run_name"], help="Naming files for this run")
parser.add_argument("--path_results", default=config.runs["path_results"], help="Folder for saving results")
parser.add_argument("--track_all_loss", default=config.runs["track_all_loss"], type=bool, help="Tracking all sub losses")
parser.add_argument("--n_epochs", default=config.runs["n_epochs"], type=int, help="Number of training epochs")
parser.add_argument("--train_model", default=config.runs["train_model"], type=bool, help="training the model")
parser.add_argument("--generate", default=config.runs["generate"], type=bool, help="Generating results")
parser.add_argument("--output_metrics", default=config.runs["output_metrics"], type=bool, help="Output ROUGE score and other metrics")


args = parser.parse_args()    
print(args)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# enable gpu if available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"The device you are using is {device}")

#######################################
######## load glove embeddings ########
glove_model = loadGloveModel(args.glove)
vocab = None

###############################################
######## Loading iterator for training ########
if args.train_model:
    logger.info(f"Loading dataloader for train")
    train_iterator, vocab = get_loader(file=args.train, 
                                       min_freq=args.min_freq, 
                                       vocab_size=args.vocab_size,
                                       remove_stopwords=args.remove_stopwords, 
                                       remove_punctuation=args.remove_punctuation,
                                       tfidf_file=args.save_tfidf_path,
                                       vectorizer_train=args.train_tfidf)
    
    logger.info(f"Loading dataloader for valid")
    valid_iterator = get_loader(file=args.valid, 
                                vocabulary=vocab, 
                                min_freq=args.min_freq, 
                                vocab_size=args.vocab_size,
                                remove_stopwords=args.remove_stopwords, 
                                remove_punctuation=args.remove_punctuation, 
                                tfidf_file = args.tfidf_file)

    if args.vocab_save:
        torch.save(vocab, args.save_vocab_path)

#################################################
######## Iterator for generating summary ########
if args.generate:
    #Using pretrained model and only generating results
    if vocab is None:
        vocab = torch.load(args.vocab_file)
    
    logger.info(f"Loading dataloader for test")
    test_iterator = get_loader(file=args.test, 
                               vocabulary=vocab, 
                               min_freq=args.min_freq, 
                               vocab_size=args.vocab_size,
                               remove_stopwords=args.remove_stopwords, 
                               remove_punctuation=args.remove_punctuation, 
                               tfidf_file = args.tfidf_file)


try:
    train_iterator
except :
    try:
        test_iterator
    except NameError :
        print('Please, set at least set variable train_model or generate to True to provide model at least one valid iterator')

##############################################
######## Intializing model parameters ########
weights_matrix = weights_matrix_embedding(glove_model, vocab.itos.values(), len(vocab), 100)
input_dim = len(vocab)
output_dim = len(vocab)

################################
######## Defining model ########
model = AE(vocab=vocab, 
           weights_matrix=weights_matrix, 
           input_dim=input_dim, 
           output_dim=output_dim, 
           enc_hid_dim=args.enc_hid_dim, 
           dec_hid_dim=args.dec_hid_dim, 
           emb_dim=args.emb_dim, 
           enc_dropout=args.enc_dropout, 
           dec_dropout=args.dec_dropout, 
           device=device, 
           use_pretrained=args.use_pretrained,
           beam_size=args.size, 
           min_dec_steps=args.min_steps, 
           num_return_seq=args.num_return_seq, 
           num_return_sum=args.num_return_sum,
           n_gram_block=args.n_gram_block).to(device)

model.apply(args.weights_init)
logger.info(f'The model has {count_parameters(model):,} trainable parameters')

###########################################
######## Keep track on tensorboard ########
if args.writer:
    writer = SummaryWriter(comment=args.run_name)
else:
    writer = None

procedure = Procedure(model=model, 
                      vocab=vocab,
                      optimizer=args.optimizer,
                      learning_rate=args.learning_rate, 
                      weight_decay=args.weight_decay, 
                      clip=args.clip,
                      lambda_=args.lambda_, 
                      device=device,
                      writer=writer,
                      include_prev=args.include_prev,
                      include_prev_epoch=args.include_prev_epoch,
                      min_seq_len=args.min_seq_len,
                      max_seq_len=args.max_seq_len,
                      max_numel_per_batch=args.max_numel_per_batch,
                      track_all_loss=args.track_all_loss,
                      fixed_len=args.fxed_len_updates)

################################
######## training model ########
if args.train_model:
    
    N_EPOCHS = args.n_epochs
    
    logger.info(f"The model will be trained for {N_EPOCHS} epochs")
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss = procedure.train(train_iterator, args.length_ratio_train, epoch, args.accumulation_steps)
        valid_loss = procedure.evaluate(valid_iterator, args.length_ratio_train, epoch)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        #If model account for novelty/coherency we start saving model when we trained with the inclusion of previous iteration
        if args.model_save:
            if args.include_prev:
                if epoch > args.include_prev_epoch:
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        torch.save(model.state_dict(), args.save_model_path)
            else:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), args.save_model_path)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\t Train Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')


###################################################
######## Generating Summary for test model ########
if args.generate: 
    if not args.train_model:
        procedure.model.load_state_dict(torch.load(args.model_file))
    
    logger.info(f"The model is generating summaries")
    
    #Getting all results
    results = procedure.summary_generation(test_iterator, args.length_ratio_test)
    src_list = results.src.to_list()
    update_ids = results.update_id.to_list()
    t_results = treat_results(results)
    
    #Creating columns name for storing final results
    col_names = []
    for i in range(len(t_results[0])):
        col_names.append('gensum_{}'.format(i))

    # Creating dataframe for storing text results
    df_textsum = pd.DataFrame(t_results, columns=col_names)
    df_textsum['source_text'] = src_list
    cols = df_textsum.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_textsum = df_textsum[cols]
    df_textsum.to_csv('{}{}.csv'.format(args.path_results, args.run_name))
    
    #Getting metrics for the batch
    if args.output_metrics:
        logger.info(f"The model is generating ROUGE scores and other metrics")
        #Getting ROUGE Score
        R_mean_results = rouge_means(results)
        cols_name = df_textsum.columns.tolist()[1:]
        
        #Getting mesuring novelty / coherency ratios
        reuse_means = []
        new_means = []
        for col in cols_name:
            reuse_means.append(pct_reuse(src_list, update_ids, df_textsum[col]))
            new_means.append(pct_new(src_list, update_ids, df_textsum[col]))
        pct_reuse_m = mean(reuse_means)
        pct_new_m = mean(new_means)
        
        #Saving metrics into csv file
        metrics_results = R_mean_results + [pct_reuse_m] + [pct_new_m]
        cols_temp = results.columns.tolist()
        rouge_name = list(filter(lambda x: x.startswith('rouge'), cols_temp))
        metrics_name = rouge_name + ['reuse_ratio'] + ['new_ratio']
        df_metrics = pd.DataFrame(metrics_results, metrics_name, columns=['value'])
        df_metrics.to_csv('{}metrics_{}.csv'.format(args.path_results, args.run_name))