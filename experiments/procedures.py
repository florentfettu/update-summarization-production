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

sys.path.append("models")
from loss import loss_estimation
from tfidf_model import tensor_tfidf

sys.path.append("experiments")
from ROUGE_eval import Rouge_eval

class Procedure():
    
    """Procedures for training and evaluating the model, and generating update summaries"""
    
    def __init__(self, model, vocab, optimizer, learning_rate, weight_decay, clip, lambda_, device, writer, include_prev, include_prev_epoch, min_seq_len, max_seq_len, max_numel_per_batch, track_all_loss, fixed_len):
        
        self.model = model
        self.vocab = vocab
        self.learning_rate = learning_rate
        self.optimizer = optimizer(params=self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        self.clip = clip
        self.lambda_ = lambda_
        self.device = device
        self.writer = writer
        self.include_prev = include_prev
        self.include_prev_epoch = include_prev_epoch
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.max_numel_per_batch = max_numel_per_batch
        self.track_all_loss = track_all_loss
        self.fixed_len = fixed_len
        self.tensor_tfidf = tensor_tfidf(self.vocab, self.lambda_)
            
    def train(self, iterator, length_ratio, epoch, accumulation_steps, disable_progress_bar=False, teacher_forcing_ratio=0.2):

        """
        Training function of the model
        Input  : iterator - batch iterator
                 length_ratio - compression ratio for constraint / summary output
                 epoch - training epoch 
                 accumulation_steps - number of iteration to accumulate gradient - to simuate fix minibatch gradient propagation
                 disable_progress_bar - tqdm display option
                 teacher_forcing_ratio
        Output : model loss per training epoch
        """
        #Initialize model training
        self.model.train()

        # Parameters
        epoch_loss = 0
        if self.track_all_loss:
            epoch_rec_loss = 0
            epoch_idf_loss = 0
            epoch_sumrec_loss = 0
        self.optimizer.zero_grad()
        
        # Training loop
        for i, (update_id, update_text, length, src_ids_ext, oovs, max_oov_len, nugget_text, tfidf_scores) in tqdm(enumerate(iterator), total=len(iterator), disable=disable_progress_bar):
            
            # Filtering too short or too long texts
            if update_text.shape[0] <= self.min_seq_len or update_text.shape[0] >= self.max_seq_len : 
                continue
            
            #Getting batch_size
            batch_size = update_text.shape[1] 
            
            # Keeping batch with a maximum of updates - outliers - defined in configs
            if batch_size > self.max_numel_per_batch: 
                continue
            
            #Intialization of summary parameters
            self.model.trg_len_summary = 0
            prev_text = torch.zeros(self.model.trg_len_summary+1,1).long().to(self.device)                    # [trg_len_summary, batch_size]
            prev_tfidf = torch.zeros(self.model.trg_len_summary+1,1).long().to(self.device)                   # [trg_len_summary, batch_size]
            prev_len = torch.tensor(self.model.trg_len_summary+1).unsqueeze(0).to(self.device)                # [batch_size] 
            
            #Tracking batch loss
            batch_loss = 0
            if self.track_all_loss:
                batch_rec_loss = 0
                batch_idf_loss = 0
                batch_sumrec_loss = 0
            
            #Iterating through batch - per update loop
            for idx in range(batch_size):
                #We need to re-initialize length to 0 for each update - otherwise we create a cumulative update summary for each update 
                if self.fixed_len:
                    self.model.trg_len_summary = 0
                src = update_text[:,idx].unsqueeze(1).to(self.device)                           # [src len, batch size]
                src_len = length[idx].unsqueeze(0).to(self.device)                              # [batch size]          
                src_ext = src_ids_ext[:,idx].unsqueeze(1).to(self.device)                       # [src len, batch size]
                max_oov_len = max_oov_len                                                       # [int]
                tfidf_tgt = tfidf_scores[idx]                                                   # dic_type
                #Defining training length of constraint length / summary | #We use non zero count because of padding of update sequence
                desired_length = int(torch.count_nonzero(src) * length_ratio)  
                
                ########### Creating tfidf tensor ###########
                if epoch > self.include_prev_epoch:
                    tfidf_tensor = self.tensor_tfidf.generate_tfidf(src, prev_tfidf, tfidf_tgt, desired_length, update=True).unsqueeze(1).to(self.device)
                else:
                    tfidf_tensor = self.tensor_tfidf.generate_tfidf(src, prev_tfidf, tfidf_tgt, desired_length).unsqueeze(1).to(self.device)
                tfidf_tensor_ext = self.tensor_tfidf.generate_ext(tfidf_tensor, src_ext)
                
                ########### Model ###########
                outputs, output_idfs, hidden_sum, hidden_src = self.model(src, src_len, prev_text, tfidf_tensor, tfidf_tensor_ext, src_ext, max_oov_len, desired_length, teacher_forcing_ratio)
                
                ########### Loss estimation ###########
                output_dim = outputs.shape[-1]
                output_idfs_dim = output_idfs.shape[-1]
                output = outputs[1:].view(-1, output_dim)                          # [seq_len -1, vocab_size]
                output_idf = output_idfs[1:].view(-1, output_idfs_dim)             # [seq_len - 1, vocab_size]
                output = torch.log(output)                                         # [(seq_len - 1) * batch size, output dim]
                output_idf = torch.log(output_idf)
                #Getting target info
                trg = src_ext[1:].view(-1)                                         # [(trg len - 1) * batch size]
                trg_idf = tfidf_tensor_ext[1:].view(-1)                            # [desired length * batch size]

                #Loss function
                loss, rec_loss, idf_loss, sum_recons = loss_estimation(output, trg, prev_text, self.vocab, update_id, idx, self.lambda_, epoch, self.include_prev_epoch, output_idf, hidden_sum, hidden_src, trg_idf)
                
                ########### Backpropagation ###########
                loss.register_hook(lambda grad: grad / batch_size)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                # Accumulating steps for computation
                if (i+1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
                ########### Generating text information for next iteration ###########
                if self.include_prev:
                    # Include the previous text information later to let the model train
                    if epoch > self.include_prev_epoch:

                        # get the highest predicted tokens from our predictions
                        prev_text = outputs.argmax(2).to(self.device)
                        prev_tfidf = output_idfs.argmax(2).to(self.device)
                        prev_len = torch.tensor(self.model.trg_len_summary).unsqueeze(0).to(self.device)
                        
            ########### Final losses estimation ###########
                batch_loss += loss.item()
                if self.track_all_loss:
                    batch_rec_loss += rec_loss.item()
                    batch_idf_loss += idf_loss.item()
                    batch_sumrec_loss += sum_recons.item()

            #final loss
            epoch_loss += (batch_loss/batch_size)
            if self.track_all_loss:
                epoch_rec_loss += (batch_rec_loss/batch_size)
                epoch_idf_loss += (batch_idf_loss/batch_size)
                epoch_sumrec_loss += (batch_sumrec_loss/batch_size)
        
        ########### Kepping track on tesnorboard ###########
        if self.writer is not None:
            self.writer.add_scalar("Train/total_loss", epoch_loss / len(iterator), global_step=epoch)
            if self.track_all_loss:
                self.writer.add_scalar("Train/rec_loss", epoch_rec_loss / len(iterator), global_step=epoch) 
                self.writer.add_scalar("Train/idf_loss", epoch_idf_loss / len(iterator), global_step=epoch)
                self.writer.add_scalar("Train/sumrec_loss", epoch_sumrec_loss / len(iterator), global_step=epoch)
        
        return epoch_loss / len(iterator)


    def evaluate(self, iterator, length_ratio, epoch=None, disable_progress_bar=False, teacher_forcing_ratio=0):
        """
        Evaluating model - no gradient backpropaagation to track loss with validation dataset
        Input and Output parameters are the same as training function 
        """
        #Initialize model evaluation
        self.model.eval()
        
        # Parameters
        epoch_loss = 0
        if self.track_all_loss:
            epoch_rec_loss = 0
            epoch_idf_loss = 0
            epoch_sumrec_loss = 0
        
        #Evaluation loop 
        with torch.no_grad():
            for i, (update_id, update_text, length, src_ids_ext, oovs, max_oov_len, nugget_text, tfidf_scores) in tqdm(enumerate(iterator), total=len(iterator), disable=disable_progress_bar):
            
                if update_text.shape[0] <= self.min_seq_len or update_text.shape[0] >= self.max_seq_len : 
                    continue
        
                batch_size = update_text.shape[1] 
                if batch_size > self.max_numel_per_batch: # on garde maximum 3 updates par batch
                    continue
                    
                #Intialization of summary parameters
                self.model.trg_len_summary = 0 
                prev_text = torch.zeros(self.model.trg_len_summary+1,1).long().to(self.device)                 # [trg_len_summary, batch_size]
                prev_len = torch.tensor(self.model.trg_len_summary+1).unsqueeze(0)                             # [batch_size] 
                prev_tfidf = torch.zeros(self.model.trg_len_summary+1,1).long().to(self.device)                # [trg_len_summary, batch_size]
                
                batch_loss = 0
                if self.track_all_loss:
                    batch_rec_loss = 0
                    batch_idf_loss = 0
                    batch_sumrec_loss = 0
            
                for idx in range(batch_size):
                    #We need to re-initialize length to 0 for each update - otherwise we create a cumulative update summary for each update 
                    if self.fixed_len:
                        self.model.trg_len_summary = 0 
                    src = update_text[:,idx].unsqueeze(1).to(self.device)                                      # [src len, batch size]
                    src_len = length[idx].unsqueeze(0).to(self.device)                                         # [batch size]          
                    src_ext = src_ids_ext[:,idx].unsqueeze(1).to(self.device)                                  # [src len, batch size]
                    max_oov_len = max_oov_len                                                                  # [int]
                    desired_length = int(torch.count_nonzero(src) * length_ratio)
                    
                    ########### Creating tfidf tensor ###########
                    tfidf_tgt = tfidf_scores[idx]
                    if epoch > self.include_prev_epoch:
                        tfidf_tensor = self.tensor_tfidf.generate_tfidf(src, prev_tfidf, tfidf_tgt, desired_length, update=True).unsqueeze(1).to(self.device)
                    else:
                        tfidf_tensor = self.tensor_tfidf.generate_tfidf(src, prev_tfidf, tfidf_tgt, desired_length).unsqueeze(1).to(self.device)
                    tfidf_tensor_ext = self.tensor_tfidf.generate_ext(tfidf_tensor, src_ext)

                    ########### Model ###########
                    outputs, output_idfs, hidden_sum, hidden_src = self.model(src, src_len, prev_text, tfidf_tensor, tfidf_tensor_ext, src_ext, max_oov_len, desired_length, teacher_forcing_ratio)  

                    ########### Loss estimation ###########
                    output_dim = outputs.shape[-1]                                # [desired_length, batch size, output_dim]
                    output_idfs_dim = output_idfs.shape[-1]
                    output = outputs[1:].view(-1, output_dim)                     # [seq_len -1, vocab_size]
                    output_idf = output_idfs[1:].view(-1, output_idfs_dim)        # [seq_len - 1, vocab_size]
                    output = torch.log(output)                                    # [desired_length * batch size, output dim]
                    output_idf = torch.log(output_idf)
                    trg = src_ext[1:].view(-1)                                    # [(trg len - 1) * batch size]
                    trg_idf = tfidf_tensor_ext[1:].view(-1)                       # [desired_length * batch size]

                    #Loss estimation                
                    loss, rec_loss, idf_loss, sum_recons = loss_estimation(output, trg, prev_text, self.vocab, update_id, idx, self.lambda_, epoch, self.include_prev_epoch, output_idf, hidden_sum, hidden_src, trg_idf)
                    
                    ########### Generating text information for next iteration ###########
                    if self.include_prev:
                        if epoch > self.include_prev_epoch:
                            prev_text = outputs.argmax(2).to(self.device)
                            prev_tfidf = output_idfs.argmax(2).to(self.device)
                            prev_len = torch.tensor(self.model.trg_len_summary).unsqueeze(0).to(self.device)
                            
                ########### Final losses estimation ###########
                    batch_loss += loss.item()
                    if self.track_all_loss:
                        batch_rec_loss += rec_loss.item()
                        batch_idf_loss += idf_loss.item()
                        batch_sumrec_loss += sum_recons.item()
                
                epoch_loss += (batch_loss/batch_size)
                if self.track_all_loss:
                    epoch_rec_loss += (batch_rec_loss/batch_size)
                    epoch_idf_loss += (batch_idf_loss/batch_size)
                    epoch_sumrec_loss += (batch_sumrec_loss/batch_size)
        
        ########### Kepping track on tesnorboard ###########
        if self.writer is not None:
            self.writer.add_scalar("Valid/total_loss", epoch_loss / len(iterator), global_step=epoch)
            if self.track_all_loss:
                self.writer.add_scalar("Valid/rec_loss", epoch_rec_loss / len(iterator), global_step=epoch) 
                self.writer.add_scalar("Valid/idf_loss", epoch_idf_loss / len(iterator), global_step=epoch)
                self.writer.add_scalar("Valid/sumrec_loss", epoch_sumrec_loss / len(iterator), global_step=epoch)

        return epoch_loss / len(iterator)

    
    def summary_generation(self, iterator, length_ratio):
        
        """
        Generating final summaries and getting evaluation metrics : ROUGE Score & %reused words
        Input  : iterator - batch iterator
                 length_ratio - compression ratio - defining final summary length
        Output : df_final - dataframe containing original text, associated summaries, and evaluation metrics.
        """
        #Intializing model for generation
        self.model.eval()
        
        #parameters and evaluation 
        rouge = Rouge_eval()  
        dict_test = []
        src_oovs_updated = []

        with torch.no_grad():
            for i, (update_id, update_text, length, src_ids_ext, oovs, max_oov_len, nugget_text, tfidf_scores) in enumerate(iterator):
                if update_text.shape[0] <= self.min_seq_len or update_text.shape[0] >= self.max_seq_len :
                    continue

                batch_size = update_text.shape[1]
                if batch_size > self.max_numel_per_batch: # on garde maximum 3 updates par batch
                    continue
                
                #Intialization of summary parameters
                self.model.trg_len_summary = 0
                summary = torch.zeros(self.model.trg_len_summary+1,1).long().to(self.device) # [trg_len_summary, batch_size]
                summary_len = torch.tensor(self.model.trg_len_summary+1).unsqueeze(0).to(self.device) # [batch_size] 
                
                #Storing iteration texts for updates
                beam_sum_ids = []
                beam_summaries = []
                
                for idx in range(batch_size):
                    #We need to re-initialize length to 0 for each update - otherwise we create a cumulative update summary for each update 
                    if self.fixed_len:
                        self.model.trg_len_summary = 0 
                    src = update_text[:,idx].unsqueeze(1).to(self.device)                  # [src len, batch size]
                    src_len = length[idx].unsqueeze(0).to(self.device)                     # [batch size]
                    src_ext = src_ids_ext[:,idx].unsqueeze(1).to(self.device)              # [src len, batch size]
                    src_oovs = oovs[idx]
                    #Creating an updated list of OOVs for update generation
                    for word in oovs[idx]:
                        if word not in src_oovs_updated:
                            src_oovs_updated.append(word) 
                    max_oov_len = max_oov_len                                              # [int]
                    update_ids = update_id[idx].item()
                    desired_length = int(torch.count_nonzero(src) * length_ratio)
                    
                    ########### Creating tfidf tensor ###########
                    tfidf_tgt = tfidf_scores[idx]
                    tfidf_tensor = [self.tensor_tfidf.generate_tfidf(src, summary_i, tfidf_tgt, desired_length, update=True).unsqueeze(1).to(self.device) for summary_i in summary]
                    tfidf_tensor_ext = [self.tensor_tfidf.generate_ext(tfidf_tensor_i, src_ext) for tfidf_tensor_i in tfidf_tensor]
                    
                    ########### Model inference - generating text summary ###########
                    sum_idx, sum_sentences, beam_sum_ids, beam_summaries = self.model.inference(src, src_len, summary, tfidf_tensor, tfidf_tensor_ext, desired_length, src_ext, max_oov_len, src_oovs, src_oovs_updated, update_ids, beam_sum_ids, beam_summaries)
                    
                    ########### Metric evaluation and storing results ###########
                    sum_sentences_eval = sum_sentences
                    beam_sum_eval = beam_summaries
                    src = re.sub("\<[a-z0-9]+\>", "", " ".join(self.vocab.invert_numericalize_extended(src_oovs, [], src_ext.squeeze(1).tolist()))).strip()
                                        
                    if update_ids == 0:
                        rouge.rouge_results([nugget_text[0][idx]], sum_sentences_eval)
                        results = rouge.final_results
                        dict_test.append((update_ids, src, nugget_text[0][idx], sum_sentences_eval, rouge.rouge_eval_inference(results)))
                    
                    else: 
                        rouge.rouge_results([nugget_text[0][idx]], beam_sum_eval)
                        results = rouge.final_results                        
                        dict_test.append((update_ids, src, nugget_text[0][idx], beam_sum_eval, rouge.rouge_eval_inference(results)))

                    ########### GEnerating texts for next iteration updates ###########
                    if update_ids == 0:
                        summary = [summary.unsqueeze(1).to(self.device) for summary in sum_idx]           # [summary_len, beam]
                    else:
                        summary = [sum_idx.permute(1,0).to(self.device) for sum_idx in beam_sum_ids]      # [summary_len, beam]
                        beam_sum_ids = []
                        beam_summaries = []

                    summary_len = torch.tensor(self.model.trg_len_summary).unsqueeze(0).to(self.device)
        
        ########################## Output final results in a dataframe ##########################
        df = pd.DataFrame(dict_test, columns = ["update_id", "src", "nugget", "generated_summaries", "rouges_score"])
        rouges_score = pd.json_normalize(df['rouges_score'])
        df.drop("rouges_score", axis=1, inplace=True)
        df_final = pd.concat((df, rouges_score), axis=1)
        
        return df_final