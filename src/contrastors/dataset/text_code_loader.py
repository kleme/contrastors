import torch.distributed
from contrastors.dataset.text_text_loader import LocalShardDataset, collate_local_ds, MAPPED_NAMES
from pathlib import Path
import webdataset as wds
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from itertools import cycle
from pyarrow.json import read_json
from tqdm import tqdm
from typing import Iterator
import json
import gzip
import logging
import pyarrow.json as pj
import dask.dataframe as dd
import torch
import random
import fsspec
import os
from contrastors.distributed import print_in_order, print_rank_zero
from webdataset.tariterators import base_plus_ext
from datasets import load_dataset, concatenate_datasets, IterableDataset
from datasets.distributed import split_dataset_by_node
from contrastors.dataset.negative_samplers import RandomSampler, TopKSampler, NoSampler, PreApplyWrapper, RandomWeightedSampler, TemperatureScheduler, BasicRepoSampler, RepoHardSampler
from contrastors.dataset.index_filter import RankFilter, RepoRankFilter, QuantileFilter

KEY2PREFIX = {"query": "query", "document": "passage", "negative": "passage"}
class RoundRobinDataLoader(Iterator):
    def __init__(self, dataloaders, epoch = 0):
        self.dataloaders = dataloaders
        
        if epoch > 0:
            [dl.sampler.set_epoch(epoch) for dl in self.dataloaders.values() if isinstance(dl, DistributedSampler)]

        self.individual_iterators = {l : iter(dl) for l, dl in self.dataloaders.items()}

        order = list(self.individual_iterators.keys())
        self.dataloader_cycle = cycle(order)
        self.cur_dataloader = order[0]
        self.finished_dataloaders = []

    def __next__(self):
        if len(self.finished_dataloaders) == len(self.individual_iterators):
            raise StopIteration

        self.cur_dataloader = next(self.dataloader_cycle)
        while self.cur_dataloader in self.finished_dataloaders:
            self.cur_dataloader = next(self.dataloader_cycle)
        try:
            return  next(
                    self.individual_iterators[self.cur_dataloader])
            
        except StopIteration:
            self.finished_dataloaders.append(self.cur_dataloader)

            if len(self.finished_dataloaders) == len(self.individual_iterators):
                raise StopIteration

            return self.__next__()
    
    def get_length(self):
        return sum(len(dl.dataset) for dl in self.dataloaders.values()) 

class ProbabilisticDataLoader(Iterator):
    def __init__(self, dataloaders, epoch = 0, probabilities = None, lengths = None):
        self.dataloaders = dataloaders
        self.yielded = {l : 0 for l in self.dataloaders.keys()}
        if lengths is None:
            self.lengths = {l: len(dl.dataset) for l, dl in self.dataloaders.items()}
        else:
            assert len(lengths) == len(self.dataloaders)
            self.lengths = lengths
        
        if probabilities is None:
            total = self.get_length()
            self.probabilities = {l : (1.0 * length)/total for l, length in self.lengths.items()}
        else:
            assert len(probabilities) == len(self.dataloaders)
            self.probabilities = probabilities
        
        if epoch > 0:
            [dl.sampler.set_epoch(epoch) for dl in self.dataloaders.values() if isinstance(dl, DistributedSampler)]

        self.individual_iterators = {l: iter(dl) for l, dl in self.dataloaders.items()}
        self.keys = list(self.dataloaders.keys())
        self.finished_dataloaders = []

    def __next__(self):
        if len(self.finished_dataloaders) == len(self.individual_iterators):
            raise StopIteration
        remaining_dataloaders = [
            l for l in self.keys if l not in self.finished_dataloaders
        ]
        remaining_probabilities = [
            self.probabilities[l] for l in remaining_dataloaders
        ]

        self.cur_dataloader = random.choices(
            remaining_dataloaders, weights=remaining_probabilities, k=1
        )[0]

        try:
            to_yield = next(self.individual_iterators[self.cur_dataloader])
            self.yielded[self.cur_dataloader] += 1
            return to_yield
        except StopIteration:
            self.finished_dataloaders.append(self.cur_dataloader)

            if len(self.finished_dataloaders) == len(self.individual_iterators):
                raise StopIteration
            
            return self.__next__()
    
    def get_length(self):
        return sum(self.lengths.values()) 
    
    def save_state(self, output_file):
        saved_state = dict(yielded = self.yielded, finished_dataloaders = self.finished_dataloaders)
        with open(output_file, 'w') as file:
            json.dump(saved_state, file, indent=4)

    def load_state(self, loaded_state):
        finished_dataloaders = loaded_state['finished_dataloaders']
        assert set(finished_dataloaders).issubset(set(self.keys))
        if set(self.keys) != set(finished_dataloaders):
            self.yielded = loaded_state['yielded']
            self.finished_dataloaders = finished_dataloaders

class DSNameAdder:
    def __init__(self, ds_name):
        self.name = ds_name
    
    def __call__(self, example):
        example['dataset_name'] = self.name
        return example

class Tok256Filter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, example):
        tokens = self.tokenizer.encode(example['query'])
        num_tokens = len(tokens)
        return num_tokens < 256
        
class CodeCollator:
    def __init__(self, tokenizer, add_prefix, neg_sampler, prefixes, temp_scheduler = None):
        self.tokenizer = tokenizer
        # self.tokenizer.padding_side = "left"
        self.add_prefix = add_prefix
        self.neg_sampler = neg_sampler
        self.prefixes = prefixes
        self.temp_scheduler = temp_scheduler

        # self.template = [
        #     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        # ]
    
    def __call__(self, batch):
        ds_name = [sample.pop("dataset_name") for sample in batch][0]
        # ds_name = "swe_repo"
        if self.temp_scheduler is not None:
            batch = self.neg_sampler(batch, self.temp_scheduler.step())
        else:
            batch = self.neg_sampler(batch)
        if batch is None:
            return {'skip_batch': True}
        keys = batch[0].keys()
        tokenized_inputs = {}
        for col in keys:
            collected = [sample[col] for sample in batch]
            if isinstance(collected[0], list):
                collected = sum(collected, [])

            if self.add_prefix and self.prefixes[ds_name].get(col, False):
                collected = [f"{self.prefixes[ds_name][col]}: {text}" for text in collected]

            # templated_messages = []
            # for text in collected:
            #     message = self.template + [{"role": "user", "content": text}]
            #     templated_messages.append(message)
                
            # chat_template = self.tokenizer.apply_chat_template(
            #     templated_messages,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )
            
            tokenized = self.tokenizer(collected, padding="max_length", truncation=True, return_tensors="pt")
            tokenized = {f"{col}_{k}": v for k, v in tokenized.items()}
            tokenized_inputs = {**tokenized_inputs, **tokenized}

        return {**tokenized_inputs, **{'dataset_name': ds_name}}


class CrossEncoderCodeCollator:
    def __init__(self, tokenizer, add_prefix, neg_sampler, prefixes, add_eos_mask = False, seq2seq = False, temp_scheduler = None, true_seq2seq_label = '▁true', false_seq2seq_label = '▁false'):
        self.tokenizer = tokenizer
        self.add_prefix = add_prefix
        self.neg_sampler = neg_sampler
        self.prefixes = prefixes
        self.add_eos_mask = add_eos_mask
        self.seq2seq = seq2seq
        self.temp_scheduler = temp_scheduler
        self.true_label = true_seq2seq_label
        self.false_label = false_seq2seq_label
    
    def __call__(self, batch):
        ds_name = [sample.pop("dataset_name") for sample in batch][0]
        if self.seq2seq:
            texts = []
            labels = []
            if 'label' in batch[0]:
                texts = [f"Query: {sample['query']} Document: {sample['document']} Relevant:" for sample in batch]
                labels = [self.true_label if sample['label'] else self.false_label for sample in batch]
            else:
                if self.temp_scheduler is not None:
                    batch = self.neg_sampler(batch, self.temp_scheduler.step())
                else:
                     batch = self.neg_sampler(batch)
                texts, labels = [], []
                for sample in batch:
                    for i, doc in enumerate(sample['document']):
                        if i == 0:
                            labels.append(self.true_label)
                        else:
                            labels.append(self.false_label)
                        texts.append(f"Query: {sample['query']} Document: {doc} Relevant:")

            tokenized = self.tokenizer(
                texts, padding=True, truncation="longest_first", return_tensors="pt", max_length=self.tokenizer.model_max_length
            )
            
            tokenized['labels'] = self.tokenizer(labels, return_tensors='pt')['input_ids']
    
            eos_mask = tokenized['input_ids'].eq(self.tokenizer.eos_token_id)
            
            if self.add_eos_mask:
                skip_batch = len(torch.unique(eos_mask.sum(1))) > 1
            else:
                skip_batch = False
            
            return {**tokenized, **{'dataset_name': ds_name, 'skip_batch': skip_batch}}

        else:
            if 'label' in batch:
                texts = [[], []]
                labels = []
                for sample in batch:
                    query = sample['query']
                    doc = sample['document']
                    labels.append(sample['label'])
                    texts[0].append(query.strip())
                    texts[1].append(doc.strip())
            else:
                if self.temp_scheduler is not None:
                    batch = self.neg_sampler(batch, self.temp_scheduler.step())
                else:
                     batch = self.neg_sampler(batch)
                texts = [[], []]
                labels = []
                for sample in batch:
                    query = sample['query']
                    for i, doc in enumerate(sample['document']):
                        if i == 0:
                            labels.append(1)
                        else:
                            labels.append(0)
                        texts[0].append(query.strip())
                        texts[1].append(doc.strip())
            tokenized = self.tokenizer(
                *texts, padding=True, truncation="longest_first", return_tensors="pt", max_length=self.tokenizer.model_max_length
            )
            labels = torch.tensor(labels, dtype=torch.float)
            eos_mask = tokenized['input_ids'].eq(self.tokenizer.eos_token_id)
            if self.add_eos_mask:
                skip_batch = len(torch.unique(eos_mask.sum(1))) > 1
            else:
                skip_batch = False
            return {**tokenized, **{'dataset_name': ds_name, 'labels': labels, 'skip_batch': skip_batch}}

def get_code_datasets(ds_spec, seed, filter_type, k = None, streaming = False, from_hub = True, filter_256 = False, tokenizer = None, 
                      num_negatives = None, abs_neg_threshold = None, neg_abs_margain = None, neg_perc_margain = None, neg_start_range = 0, neg_end_range = int(1e6), 
                      neg_sample_type = 'random', pre_apply_neg_mining = False, temperature_start = 0.5, temperature_end = 0.1, temp_decay_type = 'linear'):
    with open(ds_spec) as stream:
        spec = yaml.safe_load(stream)
        
    code_datasets = {}
    lengths = {}
    adders = {}
    filterer = {}
    prefixes = {}
    neg_samplers = {}
    for language, lang_spec in tqdm(spec.items()):
        if dist.get_rank() == 0:
            print("Started Concatenation")
        if from_hub:
            dataset = concatenate_datasets([load_dataset(ds['path'], split = 'train', 
                     streaming = streaming, num_proc=16) for ds in lang_spec["datasets"]])
        else:
            dataset = concatenate_datasets([load_dataset("json", 
                data_files=f"{ds['path']}/*.jsonl",   
                split="train", streaming=streaming) for ds in lang_spec["datasets"]])
        
        if dist.get_rank() == 0:
            print("Concatenation Done")
                
        if streaming:
            dataset = dataset.shuffle(seed = seed, buffer_size= 1000)
        else: 
            dataset = dataset.shuffle(seed=seed)
        lengths[language] = sum([ds['length'] for ds in lang_spec["datasets"]])
        
        if dist.get_rank() == 0:
            print("Shuffling Done")

        if filter_type is not None:
            print("Filter k: ", k)
            print("Filter type: ", filter_type)
            if filter_type == 'rank':
                filterer[language] = RankFilter(k, num_negatives, start_range= neg_start_range, end_range= neg_end_range, abs_margain= neg_abs_margain, perc_margain= neg_perc_margain)
            elif filter_type == 'repo-rank':
                filterer[language] = RepoRankFilter(k, num_negatives, start_range= neg_start_range, end_range= neg_end_range, abs_margain= neg_abs_margain, perc_margain= neg_perc_margain)
            elif filter_type == 'quantile':
                filterer[language] = QuantileFilter()
            # Synchronize before filtering to ensure all ranks are ready
            if not streaming:
                torch.distributed.barrier()
            if streaming:
                dataset = dataset.filter(filterer[language])
            else:
                # run on rank 0 first to avoid duplication
                if dist.get_rank() % dist.get_world_size() == 0:
                    dataset = dataset.filter(filterer[language], num_proc=32)
                torch.distributed.barrier()
                if dist.get_rank() % dist.get_world_size() != 0:
                    dataset = dataset.filter(filterer[language], num_proc=32)
                torch.distributed.barrier()
        
        if filter_256 and tokenizer is not None:
            filterer2 = Tok256Filter(tokenizer)
            if not streaming:
                torch.distributed.barrier()
            if streaming:
                dataset = dataset.filter(filterer2)
            else:
                if dist.get_rank() % dist.get_world_size() == 0:
                    dataset = dataset.filter(filterer2, num_proc=32)
                torch.distributed.barrier()
                if dist.get_rank() % dist.get_world_size() != 0:
                    dataset = dataset.filter(filterer2, num_proc=32)
                torch.distributed.barrier()

        if dist.get_rank() == 0:
            print("Filtering Done")
        
        dataset = dataset.take(min(len(dataset), lengths[language])) if not streaming else dataset.take(lengths[language])
        
        for ds in lang_spec["datasets"]:
            prefixes[language] = {}
            if ds.get('query_prefix', None) is not None:
                prefixes[language]['query'] = ds['query_prefix']
            if ds.get('document_prefix', None) is not None:
                prefixes[language]['document'] = ds['document_prefix']
            break
        adders[language] = DSNameAdder(language)
        
        if not streaming:
            torch.distributed.barrier()
        
        if dist.get_rank() == 0:
            print("Started Mapping")

        if streaming:
            dataset = dataset.map(adders[language])
        else:
            if dist.get_rank() % dist.get_world_size() == 0:
                dataset = dataset.map(adders[language], num_proc=32)
            torch.distributed.barrier()
            if dist.get_rank() % dist.get_world_size() != 0:
                dataset = dataset.map(adders[language], num_proc=32)
            torch.distributed.barrier()

        # dataset = dataset.add_column("dataset_name", [language] * len(dataset))

        if dist.get_rank() == 0:
            print("Mapping Done")
        
        if not streaming:
            torch.distributed.barrier()
        
        # dataset = split_dataset_by_node(dataset, rank=dist.get_rank(), 
        #                     world_size=dist.get_world_size())
        
        if dist.get_rank() == 0:
            print("Splitting Done")
        
        if pre_apply_neg_mining:
            lengths[language] *= (num_negatives + 1)
        
        if neg_sample_type == 'random':
            neg_sampler = RandomSampler(num_negatives, start_range= neg_start_range, end_range= neg_end_range, abs_margain= neg_abs_margain, perc_margain= neg_perc_margain)
        elif neg_sample_type == 'topk':
            neg_sampler = TopKSampler(num_negatives, neg_start_range, neg_end_range, abs_neg_threshold)
        elif neg_sample_type =='random-weighted':
            # TODO: try to add perc_margain and sample from those
            neg_sampler = RandomWeightedSampler(num_negatives, start_range= neg_start_range, end_range= neg_end_range, abs_margain= neg_abs_margain, perc_margain= neg_perc_margain)
        elif neg_sample_type == 'repo-hard':
            neg_sampler = RepoHardSampler(num_negatives, start_range= neg_start_range, end_range= neg_end_range, abs_margain= neg_abs_margain, perc_margain= neg_perc_margain)
        elif neg_sample_type in ('repo-simple', 'repo-random'):
            neg_sampler = BasicRepoSampler(num_negatives, random_negs= neg_sample_type == 'repo-random')
        else:
            neg_sampler = NoSampler()
        if pre_apply_neg_mining:
            neg_sampler.pre_apply = True
            pre_apply_wrapper = PreApplyWrapper(neg_sampler)
            cols_to_remove = ['metadata', 'negatives', 'negative_scores', 'document_score', 'document_rank']
            if streaming:
                dataset = dataset.map(pre_apply_wrapper, batched = True, batch_size = 1, remove_columns= cols_to_remove)
            else:
                dataset = dataset.map(pre_apply_wrapper, batched = True, batch_size = 1, num_proc= 8, remove_columns= cols_to_remove)
            neg_sampler = NoSampler()
        code_datasets[language] = dataset 
        neg_samplers[language] = neg_sampler
    
    if neg_sample_type =='random-weighted':
        temp_scheduler = TemperatureScheduler(temperature_start, temperature_end, sum(lengths.values()), decay_type= temp_decay_type)
    else:
        temp_scheduler = None
    return (code_datasets, lengths, prefixes, neg_samplers, temp_scheduler)
    
def get_iterable_code_dataloader(code_datasets, lengths, prefixes, neg_samplers, temp_scheduler, batch_size, tokenizer, seed, add_prefix, num_workers=0, epoch=0, streaming = False, 
                                  is_cross_encoder = False, add_eos_mask = False, seq2seq = False, yielded = None, global_batch_size = None):
    dataloaders = {}
    lengths_epoch = {}
    total = sum(lengths.values())
    probabilities = {l : (1.0 * length)/total for l, length in lengths.items()}
    if yielded is not None:
        [temp_scheduler.step() for _ in range(sum(yielded.values()) * global_batch_size)]
    elif temp_scheduler is not None:
        temp_scheduler.reset()

    world_size = dist.get_world_size()

    for language, dataset in code_datasets.items():
        if streaming: 
            dataset = dataset.shuffle(seed= seed + epoch, buffer_size=1000)
        else:
            dataset = dataset.shuffle(seed + epoch)
        
        if is_cross_encoder:
            collate_fn = CrossEncoderCodeCollator(tokenizer, add_prefix=add_prefix, neg_sampler= neg_samplers[language], prefixes = prefixes, add_eos_mask= add_eos_mask, seq2seq= seq2seq, temp_scheduler= temp_scheduler)
        else:
            collate_fn = CodeCollator(tokenizer, add_prefix=add_prefix, neg_sampler= neg_samplers[language], prefixes = prefixes, temp_scheduler= temp_scheduler)
        
        if yielded is not None:
            skip_samples = yielded[language] * global_batch_size
            total_samples = lengths[language] - skip_samples
            # Ensure total samples is divisible by global batch size
            total_samples = (total_samples // global_batch_size) * global_batch_size
            ds = split_dataset_by_node(dataset.skip(skip_samples).take(total_samples), rank=dist.get_rank(), world_size=world_size)
            lengths_epoch[language] = total_samples
        else:
            total_samples = (lengths[language] // global_batch_size) * global_batch_size
            ds = split_dataset_by_node(dataset.take(total_samples), rank=dist.get_rank(), world_size=world_size)
            lengths_epoch[language] = total_samples
        
        nw = min(num_workers, dataset.n_shards) if streaming else num_workers
        dataloaders[language] = DataLoader(
            ds, batch_size=batch_size,
            collate_fn=collate_fn, drop_last=False, num_workers=nw
        )
    return ProbabilisticDataLoader(dataloaders, epoch, probabilities= probabilities, lengths = lengths_epoch)