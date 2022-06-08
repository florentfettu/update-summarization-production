import json
import logging
import os
import pdb
from collections import defaultdict
from copy import copy
from pprint import pprint
import statistics
import numpy as np
import spacy
from nltk.corpus import stopwords
from rouge_score import rouge_scorer


class Rouge_eval(object):

    def __init__(self, remove_stopwords=True, use_stemmer=True):
        
        self.remove_stopwords = remove_stopwords
        
        if remove_stopwords:
            self.stopwords = set(stopwords.words("english"))
            
        self.use_stemmer = use_stemmer
        
        self.nlp = spacy.load("en_core_web_sm")
        
        # python implementation of ROUGE
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                            use_stemmer=use_stemmer)
        
        #Create dictionnary for storing final scores
        self.final_results = {}
    
    def calc_rouges(self, ref, summ):
        
        """Estimating ROUGE scores between summary and provided reference"""
        
        if self.remove_stopwords:
            ref = ' '.join([str(word) for word in self.nlp(str(ref)) if str(word) not in self.stopwords])
            summ = ' '.join([str(word) for word in self.nlp(str(summ)) if str(word) not in self.stopwords])
        
        rouge_scores = self.rouge_scorer.score(ref, summ)
        
        return rouge_scores
        
    def get_rouge_defaultdict(self, default_type=float, store_summ=False):
        
        """Dictionnary for storing different scores"""
        
        if store_summ:
            dict_ = {'text' : None, 
                    'rouge1': defaultdict(default_type),
                    'rouge2': defaultdict(default_type),
                    'rougeL': defaultdict(default_type)}
        else:
            dict_ = {'rouge1': defaultdict(default_type),
                     'rouge2': defaultdict(default_type),
                     'rougeL': defaultdict(default_type)}
        return dict_
    
    def rouge_results(self, references, generated_summaries, list_metric=['precision', 'recall', 'fmeasure']):
        
        """ROUGE score between the reference and all generated summaries (beam_size). Here since there is only one reference, there is no difference between avg; max, or min in dictionnary"""
        
        # Create dictionnary that is going to store results for the group
        rouges = {}
        batch_result = {}
        
        # We go through the 3 summaries generated by the model - 1 output by topic
        for i, summ in enumerate(generated_summaries):
            rouges[i] = self.get_rouge_defaultdict(default_type=list, store_summ=True)
            rouges[i]['text'] = summ
            for ref in references:
                rouge_scores = self.calc_rouges(ref, summ)
                for rouge_name, rouge_obj in rouge_scores.items():
                    for metric in list_metric:
                        score = getattr(rouge_obj, metric)
                        rouges[i][rouge_name][metric].append(score)
                        
                        
            # Compute statistics for each summary
            avg_rouges = self.get_rouge_defaultdict()
            min_rouges = self.get_rouge_defaultdict()
            max_rouges = self.get_rouge_defaultdict()
            std_rouges = self.get_rouge_defaultdict()
            for rouge_name, rouge_obj in rouges[i].items():
                if rouge_name == 'text':
                    pass
                else:
                    for metric in list_metric:
                        scores = rouges[i][rouge_name][metric]
                        
                        avg_, min_, max_, std_ = np.mean(scores), np.min(scores), np.max(scores), np.std(scores)
                        avg_rouges[rouge_name][metric] = avg_
                        min_rouges[rouge_name][metric] = min_
                        max_rouges[rouge_name][metric] = max_
                        std_rouges[rouge_name][metric] = std_
            

            batch_result['gensumm_{}'.format(i)] = {'text': summ, 'references': references, 
                                            'avg': avg_rouges, 'min': min_rouges, 
                                            'max' : max_rouges, 'std' : std_rouges}    
        
        self.final_results = batch_result

    def rouge_eval_inference(self, results):
        
        """Getting final results - estimating average ROUGE score between all beams/summaries generated"""
        
        final_results = self.get_rouge_defaultdict(default_type=list)
        
        rouge1p = []
        rouge1r = []
        rouge1f = []
        rouge2p = []
        rouge2r = []
        rouge2f = []
        rougeLp = []
        rougeLr = []
        rougeLf = []
        
        #  [{},{},{},{},{}]  beam size dictionaries
        for key, value in results.items(): 
            
            #Getting ROUGE score for all beam solutions 
             # We take the dictionnary because we don't care it's the same values! #{rouge1 : {prec, recall, fmea,}, rouge2: same, rougeLsame}
            gen_sum_dic = value["avg"]                            
            rouge1p.append(gen_sum_dic["rouge1"]["precision"])
            rouge1r.append(gen_sum_dic["rouge1"]["recall"])
            rouge1f.append(gen_sum_dic["rouge1"]["fmeasure"])
            rouge2p.append(gen_sum_dic["rouge2"]["precision"])
            rouge2r.append(gen_sum_dic["rouge2"]["recall"])
            rouge2f.append(gen_sum_dic["rouge2"]["fmeasure"])
            rougeLp.append(gen_sum_dic["rougeL"]["precision"])
            rougeLr.append(gen_sum_dic["rougeL"]["recall"])
            rougeLf.append(gen_sum_dic["rougeL"]["fmeasure"])
            
        #We estimate the mean value of all beam in the results
        final_results["rouge1"]["precision"] = statistics.mean(rouge1p)
        final_results["rouge1"]["recall"] = statistics.mean(rouge1r)
        final_results["rouge1"]["fmeasure"] = statistics.mean(rouge1f)
        final_results["rouge2"]["precision"] = statistics.mean(rouge2p)
        final_results["rouge2"]["recall"] = statistics.mean(rouge2r)
        final_results["rouge2"]["fmeasure"] = statistics.mean(rouge2f)
        final_results["rougeL"]["precision"] = statistics.mean(rougeLp)
        final_results["rougeL"]["recall"] = statistics.mean(rougeLr)
        final_results["rougeL"]["fmeasure"] = statistics.mean(rougeLf)

        return final_results