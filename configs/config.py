import torch
import sys
sys.path.append("experiments")
from run_utils import normal_weights_init, xavier_weights_init, orthogonal_weights_init

#Path for loading the necessary files for using the model
#Data parameters
data = {
    "glove" : "data/embeddings/glove.6B.100d.txt",
    "train": "data/files/trec_train_doc.csv",
    "valid": "data/files/trec_valid_doc.csv",                           #path with data use in paper
    "test": "data/files/trec_test_doc.csv",
    "min_freq" : 2,                                                     # Minimum number of word occurence to be included in Vocabulary and tfidf calculations
    "vocab_size": 10000,                                                # Vocab size - if None no limitation and All vocab will be included
    "remove_stopwords": False,                   
    "remove_punctuation": True,
}

#Summarization parameters
sum_param = {
    "length_ratio_train": 0.4,                   # must be 0 < ratio < 1 - compression ratio during training
    "length_ratio_test": 0.4,                    # must be 0 < ratio < 1 - compression ratio during generation
    "lambda_": 0.3,                               # must be -0.5 < lambda < 1.5 - otherwise instable results due to adversarial learning objective
}


#Runs experiment:
runs = {
    "writer":False,
    "run_name":"default_run",
    "path_results": "experiments/data_results/",
    "track_all_loss":False,
    "n_epochs" : 80,
    "train_model" : True,
    "generate": True,
    "output_metrics": True,
}

#Data constraints for trainer
trainer = {
    "save_tfidf_path": "experiments/model_save/{}/tfidf.pk".format(runs['run_name']),     # path to the pretrained tfidf constraint model
    "save_model_path" : "experiments/model_save/{}/best-model.pt".format(runs['run_name']),
    "save_vocab_path" : "experiments/model_save/{}/vocab.pth".format(runs['run_name']),
    "model_file" : "experiments/model_save/{}/best-model.pt".format(runs['run_name']),
    "vocab_file" : "experiments/model_save/{}/vocab.pth".format(runs['run_name']),
    "tfidf_file" : "experiments/model_save/{}/tfidf.pk".format(runs['run_name']),
    "min_seq_len": 5,                                                       # Minimum length of sentence
    "max_seq_len": 100,                                                     # Maximum length of sentence
    "max_numel_per_batch": 5,                                               # Number of update taken into account
    "accumulation_steps": 32,                                               # Batch size for gradient descent learning
    "vocab_save" : True,
    "model_save" : True,
    "train_tfidf":True,                                                 #If true train model on dataset. If false, use a presaved model at tfidf_file path
}

model_parameters = {
        "emb_dim": 100,
        "enc_hid_dim": 512, 
        "dec_hid_dim": 512, 
        "enc_dropout": 0.2,
        "dec_dropout": 0.2,
        "use_pretrained": True,
        "fxed_len_updates":True,
        "weights_init": xavier_weights_init
}

#Training parameters for the model - optimized with optuna on TREC dataset
model_hyperparameters = {
        "optimizer": torch.optim.Adam,
        "learning_rate": 1e-4,
        "weight_decay": 8e-3,
        "clip": 10,
        "include_prev": True,
        "include_prev_epoch": 40 
}

#Generation parameters with beam search
beam = {
    "size": 5,
    "min_steps": 5,                            # Minimum sequence length to generate before taking into account <eos>
    "num_return_seq": 5,
    "num_return_sum": 5,
    "n_gram_block": 3
}