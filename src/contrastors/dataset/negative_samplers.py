import random 
import torch
import torch.nn.functional as F
import numpy as np

class TemperatureScheduler:
    def __init__(self, temperature_start, temperature_end, n_steps, decay_type='linear'):
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.n_steps = n_steps
        self.decay_type = decay_type.lower()
        self.temperatures = iter(self._get_temperature_schedule())

    def _get_temperature_schedule(self):
        if self.decay_type == 'linear':
            return np.linspace(self.temperature_start, self.temperature_end, self.n_steps)
        elif self.decay_type == 'exponential':
            decay_rate = np.log(self.temperature_start / self.temperature_end) / (self.n_steps - 1)
            return self.temperature_start * np.exp(-decay_rate * np.arange(self.n_steps))
        else:
            raise ValueError(f"Unsupported decay type: {self.decay_type}. Choose from 'linear', 'exponential'.")
    
    def reset(self):
        self.temperatures = iter(self._get_temperature_schedule())
    def step(self):
        try:
            return next(self.temperatures)
        except StopIteration:
            return self.temperature_end

class PreApplyWrapper:
    def __init__(self, neg_sampler):
        self.neg_sampler = neg_sampler
    
    def __call__(self, batch):
        batch = {k : v[0] for k, v in batch.items()}
        processed_batch = self.neg_sampler(batch)
        labels = []
        documents = []
        for i, doc in enumerate(processed_batch['document']):
            if i == 0:
                labels.append(1)
            else:
                labels.append(0)
            documents.append(doc)
        
        dct = {k : [v] * len(documents) for k, v in processed_batch.items() if k != 'document'}
        dct['document'] = documents 
        dct['label'] = labels 
        return dct
        
   
class NoSampler:
    def __init__(self):
        pass 
    
    def __call__(self, batch):
        for ex in batch:
            negative_scores = ex.pop('negative_scores', None)
            document_score = ex.pop('document_score', None)
            ex.pop('document_rank', None)
            ex.pop('metadata', None)
            negatives = ex.pop('negatives', None)
        return batch

class RandomSampler:
    def __init__(self, num_negatives, start_range = 0, end_range = int(1e6), abs_margain = None, perc_margain = None, pre_apply = False):
        self.num_negatives = num_negatives
        self.abs_margain = abs_margain
        self.perc_margain = perc_margain
        self.start_range = start_range
        self.end_range = end_range
        self.pre_apply = pre_apply

    def __call__(self, batch):
        if self.pre_apply: 
            batch = [batch]
        processed_batch = []
        for ex in batch:
            negative_scores = ex.pop('negative_scores', None)
            document_score = ex.pop('document_score', None)
            ex.pop('document_rank', None)
            ex.pop('metadata', None)
            negatives = ex.pop('negatives')
            sampled = self._select_negs(negatives, negative_scores, document_score)
            if sampled is None:
                return None
            ex['document'] = [ex['document']] + sampled

            processed_batch.append(ex)
        return processed_batch[0] if self.pre_apply else processed_batch
    
    def _select_negs(self, negatives, negative_scores, document_score):
        
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
        if len(filtered_negatives) >= self.num_negatives:
            sampled = random.sample(filtered_negatives, self.num_negatives)
        else:
            # Handle the case when there are not enough items
            return None
            sampled = filtered_negatives[:self.num_negatives]

        return sampled

class TopKSampler:
    def __init__(self, num_negatives, neg_start_range, neg_end_range, neg_threshold = None):
        self.num_negatives = num_negatives
        self.neg_start_range = neg_start_range
        self.neg_end_range = neg_end_range
        self.neg_threshold = neg_threshold

    def __call__(self, batch):
        processed_batch = []
        for ex in batch:
            negative_scores = ex.pop('negative_scores', None)
            ex.pop('document_score', None)
            ex.pop('document_rank', None)
            ex.pop('metadata', None)
            negatives = ex.pop('negatives', None)
            sampled = self._select_negs(negatives, negative_scores)
            ex['document'] = [ex['document']] + sampled

            processed_batch.append(ex)
        return processed_batch

    def _select_negs(self, negatives, negative_scores):
        i = self.neg_start_range
        selected = []
        while len(selected) < self.num_negatives and i < min(len(negatives), self.neg_end_range):
            if self.neg_threshold is None or float(negative_scores[i]) <= self.neg_threshold:
                selected.append(negatives[i])
            i += 1
        return selected 



class RandomWeightedSampler:
    def __init__(self, num_negatives, start_range = 0, end_range = int(1e6), abs_margain = None, perc_margain = None, pre_apply = False):
        self.num_negatives = num_negatives
        self.abs_margain = abs_margain
        self.perc_margain = perc_margain
        self.start_range = start_range
        self.end_range = end_range
        self.pre_apply = pre_apply

    def __call__(self, batch, temperature = 0.1):
        if self.pre_apply: 
            batch = [batch]
        processed_batch = []
        for ex in batch:
            negative_scores = ex.pop('negative_scores', None)
            document_score = ex.pop('document_score', None)
            ex.pop('document_rank', None)
            ex.pop('metadata', None)
            negatives = ex.pop('negatives')
            sampled = self._select_negs(negatives, negative_scores, document_score, temperature)
            if sampled is None:
                return None
            ex['document'] = [ex['document']] + sampled

            processed_batch.append(ex)
        return processed_batch[0] if self.pre_apply else processed_batch
    
    def _select_negs(self, negatives, negative_scores, document_score, temperature):
        if negative_scores is not None and document_score is not None:
            if self.abs_margain is not None:
                filtered_negatives, filtered_scores = [], []
                for neg, score in zip(negatives, negative_scores):
                    if float(score) < float(document_score) - self.abs_margain:
                        filtered_negatives.append(neg)
                        filtered_scores.append(score)
                
                
                
            elif self.perc_margain is not None:
                filtered_negatives, filtered_scores = [], []
                for neg, score in zip(negatives, negative_scores):
                    if float(score) < float(document_score) * self.perc_margain:
                        filtered_negatives.append(neg)
                        filtered_scores.append(float(score))
            else:
                filtered_negatives, filtered_scores = negatives, [float(score) for score in negative_scores]
        else:
                filtered_negatives, filtered_scores = negatives, [float(score) for score in negative_scores]
        
        filtered_negatives = filtered_negatives[self.start_range : self.end_range]
        filtered_scores = filtered_scores[self.start_range : self.end_range]
        
        if len(filtered_negatives) >= self.num_negatives:
            sampled = self.soft_curriculum_sample(filtered_negatives, filtered_scores, self.num_negatives, temperature)
        
        else:
            # Handle the case when there are not enough items
            return None
            sampled = filtered_negatives[:self.num_negatives]

        return sampled
    
    def soft_curriculum_sample(self, filtered_negatives, filtered_scores, num_negatives, temperature):
        soft_scores = F.softmax(torch.tensor(filtered_scores) / temperature, dim=-1).tolist()
        soft_scores = np.array(soft_scores) / np.sum(soft_scores)
        selected_idxs = np.random.choice(list(range(len(filtered_negatives))), 
                                         size=num_negatives,replace=False, p=soft_scores)
        return [filtered_negatives[i] for i in selected_idxs]