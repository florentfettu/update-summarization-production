# update-summarization

Autoencoder model for update text summarization as described in Carichon. F, Fettu, F, et al. (2022) - Unsupervised Update Summarization of News Events. 

## Description

The model proposes a semi-abstractive summarization algorithm based on an autoencoder architecture. The model modify the classic autoencoder structure by adding and update function in both reconstruction and information constraint objective functions.

## Install

Python version = 3.6 at least
Run the requirements.txt to install all necessary libraries

## How tu run model :

1. All parameters of the model can be modified in the "configs/config.py" file.
2. To train the model : 
    a. Set the train_model variable to True
    b. Input a run_name (str) to record all results and model
3. To generate summaries and metrics, the variable generate must be set to True


## Dependency Management & Reproducibility

To complete 

## Project Organization

```
├── README.md               <- README for developers.
├── requirements.txt        <- Python libraries needed for running model.
├── configs                 <- Directory for configurations of model & application.
├── data                    <- Data specifics.
│   ├── embeddings          <- Storing pretrained embeddings models -- Glove.
│   ├── files               <- Train/valid/Test data of TREC 2013/2014/2015 as in paper.
│   ├── vectorizers         <- Storing pretrained tfidf/information contraint model.
├── experiments             <- Trained and serialized models, model predictions,
│                              run metrics, or model summaries.
│   ├── data_results        <- Storing generated summaries and metrics.
│   ├── model_save          <- Storing pretrained save model after training.
├── models                  <- Unsupervised Update Autoencoder for summarization.
├── runs                    <- Tensorboard runs.
├── main.py                 <- Runing code for defined configs.
```
