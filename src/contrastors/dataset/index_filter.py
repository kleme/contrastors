

from typing import Any
import numpy as np

class RepoRankFilter:
    def __init__(self, k, num_negatives, start_range = 0, end_range = int(1e6), abs_margain = None, perc_margain = None, pre_apply = False):
        self.k = k 
        self.num_negatives = num_negatives
        self.abs_margain = abs_margain
        self.perc_margain = perc_margain
        self.start_range = start_range
        self.end_range = end_range
        self.pre_apply = pre_apply    
    def __call__(self, example):
        rank = int(example['positive_code_rank']) + 1
        print("Rank: ", rank)
        if not (1 <= rank <= self.k):
            return False 
        return self.less_negs(example)
    
    def less_negs(self, example):
        negative_scores = example.get('negative_code_scores', None)
        negatives = example.get('negative_codes')
        document_score = example.get('positive_code_score', None)
        if negative_scores is not None and document_score is not None:
            if self.abs_margain is not None:
                filtered_negatives = [neg for neg, score in zip(negatives, negative_scores) if float(score) < float(document_score) - self.abs_margain]
            elif self.perc_margain is not None:
                filtered_negatives = [neg for neg, score in zip(negatives, negative_scores) if float(score) < float(document_score) * self.perc_margain]
            else:
                filtered_negatives = negatives
        else:
                filtered_negatives = negatives
        filtered_negatives = filtered_negatives[self.start_range : self.end_range] 
        return len(filtered_negatives) >= self.num_negatives   

class RankFilter:
    def __init__(self, k, num_negatives, start_range = 0, end_range = int(1e6), abs_margain = None, perc_margain = None, pre_apply = False):
        self.k = k 
        self.num_negatives = num_negatives
        self.abs_margain = abs_margain
        self.perc_margain = perc_margain
        self.start_range = start_range
        self.end_range = end_range
        self.pre_apply = pre_apply    
    def __call__(self, example):
        rank = int(example['document_rank']) + 1
        if not (1 <= rank <= self.k):
            return False 
        return self.less_negs(example)
    
    def less_negs(self, example):
        negative_scores = example.get('negative_scores', None)
        negatives = example.get('negatives')
        document_score = example.get('document_score', None)
        if negative_scores is not None and document_score is not None:
            if self.abs_margain is not None:
                filtered_negatives = [neg for neg, score in zip(negatives, negative_scores) if float(score) < float(document_score) - self.abs_margain]
            elif self.perc_margain is not None:
                filtered_negatives = [neg for neg, score in zip(negatives, negative_scores) if float(score) < float(document_score) * self.perc_margain]
            else:
                filtered_negatives = negatives
        else:
                filtered_negatives = negatives
        filtered_negatives = filtered_negatives[self.start_range : self.end_range] 
        return len(filtered_negatives) >= self.num_negatives       

class QuantileFilter:
    def __init__(self, quantile = 0.9):
        self.quantile = quantile
    
    def __call__(self, example):
        scores = [example['document_score'], *example['negative_scores']]
        scores = [float(score) for score in scores]
        threshold = np.quantile(scores, self.quantile)
        return scores[0] >= threshold

