from huggingface_hub import hf_hub_download
from pathlib import Path
import pandas as pd
import os
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm 
import json 
import gzip
from torch.utils.data import DataLoader
import webdataset as wds
import fsspec
from argparse import ArgumentParser

CSN_LANGUAGES = ['go', 'python', 'java', 'javascript', 'php', 'ruby']
CONTRASTIVE_CODE_DIR = "/tmp/contrastive-code/"
S3_COMMAND = "pipe: aws s3 cp --endpoint-url https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ --cli-read-timeout=300 {s3_uri} -"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_type", choices= ['csn_raw', 'csn_nomic_pretrained', 'cosqa'], required=True)
    return parser.parse_args()
    
def csn_loader_raw(language):
    cache_dir = Path.home() / ".cache"
    path = Path(f'{cache_dir}/huggingface/hub/datasets--code-search-net--code_search_net/snapshots/fdc6a9e39575768c27eb8a2a5f702bf846eb4759/data/{language}/{language}/final/jsonl/train/')
    assert path.exists(), f"CSN data does not exist for language {language}"
    dataset = concatenate_datasets([load_dataset('json', data_files=str(file), streaming = False, split='train') for file in sorted(path.glob('*.jsonl.gz'))])
    
    #TODO: we can add the other columns too for more advanced filtering later
    dataset = dataset.select_columns(['docstring', 'code']).rename_column('docstring', 'query').rename_column('code', 'document')
    return dataset

def csn_loader_nomic_pretrained():
    fs = fsspec.filesystem("s3", config_kwargs={"connect_timeout": 600, "read_timeout": 600})
    urls = wds.shardlists.expand_urls("s3://contrastive-index-filtered/codesearch_full/shard-{00000..00008}.jsonl.gz")
    for url in urls:
        local_path = Path(f"{CONTRASTIVE_CODE_DIR}/codesearch_full/{os.path.basename(url)}")
        if not local_path.parent.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            fs.get(url, str(local_path))
    

if __name__ == "__main__":
    args = parse_args()
    
    if args.dataset_type == 'cosqa':
        with open('/tmp/code_eval/nl2code/cosqa/cosqa-retrieval-train-19604.json', 'r') as file:
            dataset = json.load(file)
       
        dataset = [{'query': dat['docstring_tokens'], 'document': dat['code_tokens']} for dat in dataset]
        
        output_dir = Path('/tmp/contrastive-code/cosqa_full')
        output_dir.mkdir(parents = True, exist_ok= True)
        shard_size = 100_000
        for shard_start in tqdm(range(0, len(dataset), shard_size), desc="Writing shards"):
            dataset_slice = dataset[shard_start : shard_start + shard_size]
            shard_num = shard_start // shard_size
            with gzip.open(output_dir / f"shard-{shard_num:05d}.jsonl.gz", "wt") as f:
                for data in tqdm(dataset_slice, desc=f"Writing shard {shard_num:05d}"):
                    f.write(json.dumps(data) + "\n")
        
    
    elif args.dataset_type == 'csn_raw':
        for language in CSN_LANGUAGES:
            csn_dataset = csn_loader_raw(language)
            
            dataloader = DataLoader(csn_dataset, 
                                    batch_size= 100000, 
                                    num_workers= 0, 
                                    persistent_workers= False,
                                    collate_fn= None, 
                                    drop_last= False)
            
            output_dir = Path(f"{CONTRASTIVE_CODE_DIR}/code_search_net/{language}")
            
            if not output_dir.exists():
                os.makedirs(output_dir)
            
            for shard_num, dataset_slice in tqdm(enumerate(dataloader), desc="Writing shards"):
                with gzip.open(output_dir / f"shard-{shard_num:05d}.jsonl.gz", "wt") as f:
                    for data in tqdm([dict(zip(dataset_slice, v)) for v in zip(*dataset_slice.values())], desc=f"Writing shard {shard_num:05d}"):
                        f.write(json.dumps(data) + "\n")
    
    
    elif args.dataset_type == 'csn_nomic_pretrained':
        csn_loader_nomic_pretrained()   
        












