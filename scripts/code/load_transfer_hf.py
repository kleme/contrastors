from pathlib import Path
from tqdm import tqdm 
import gzip
import json
from datasets import Dataset
from datasets import load_dataset, concatenate_datasets
import os 
from torch.utils.data import DataLoader
import fire 
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_LOC = 'src/contrastors/data/'

LANGUAGES = {'go': 70,  
             'java': 160,  
             'javascript': 24,  
             'php': 30,  
             'python': 100, 
             'ruby': 9}


def write_shard(output_dir, shard_num, dataset_slice):
    shard_file = output_dir / f"shard-{shard_num:05d}.jsonl.gz"
    with gzip.open(shard_file, "wt") as f:
        for data in dataset_slice:
            f.write(json.dumps(data) + "\n")

def store_shards(ds, output_path, num_workers= 36):
    output_dir = Path(output_path)
    
    if not output_dir.exists():
        os.makedirs(output_dir)
    
    shard_size = 100_000
    shard_tasks = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for shard_num, shard_start in enumerate(range(0, len(ds), shard_size)):
            dataset_slice = ds.select(range(shard_start, min(shard_start + shard_size, len(ds))))
            shard_tasks.append(executor.submit(write_shard, output_dir, shard_num, dataset_slice))
        
        for future in tqdm(as_completed(shard_tasks), total=len(shard_tasks), desc="Writing shards"):
            future.result()
 
def load_from_hf(**kwargs):
    ds = load_dataset(f'nomic-uiuc/cosqa-mined-v1', split = 'train')
    store_shards(ds, f'{DATA_LOC}/cosqa_full_mined/')
    
    ds = load_dataset(f'nomic-uiuc/codesearch-mined-v1', split = 'train')
    store_shards(ds, f'{DATA_LOC}/codesearch_full_mined/')
    
    for lang in LANGUAGES.keys():
        ds = load_dataset(f'nomic-uiuc/stack-mined-{lang}-v1', split = 'train')
        store_shards(ds, f'{DATA_LOC}/stack_v0_most_mined/{lang}')

def load_pre_hf(**kwargs):
    for lang, iter in [('Python', 10), ('JavaScript', 18), ('Java', 25), ('Go', 2), ('Ruby', 3), ('PHP', 8)]:
        ds_lst = []
        for i in range(iter):
            try:
                ds_lst.append(load_dataset(f'nomic-uiuc/the-stack-v2-dedup-{lang}-{i}-processed', split = 'train'))
            except:
                continue
        
        store_shards(concatenate_datasets(ds_lst), f'{DATA_LOC}/stack_v0_prefilter/{lang.lower()}')
        subprocess.run(['rm', '-r', '/home/tarun/.cache/huggingface/datasets'])

def transfer_shards_custom(dir_name, save_name, **kwargs):
    path = Path(dir_name)
    
    files = []
    if path.exists() and path.is_dir():
            files.extend([(f, str(f.parent.stem)) for f in sorted(path.glob("shard-*.jsonl.gz"))])
    
    ds = None
    for file, dataset_name in tqdm(files, desc="Loading files"):
        if file.suffix == ".gz":
            filehandler = gzip.open(file, "rt")
        else:
            filehandler = open(file, "r")
        
        
        with filehandler as f:
            for line in f:
                data = json.loads(line)
                if ds is None:
                    ds = {k : [] for k in data.keys()}
                
                for k, v in data.items():
                    ds[k].append(v)
    
    ds = Dataset.from_dict(ds)
    ds.push_to_hub(save_name, private= True)

def transfer_shards(**kwargs):
    for lang in LANGUAGES.keys():
        path = Path(f'{DATA_LOC}/stack_v0_most_mined/{lang}')
        files = []
        if path.exists() and path.is_dir():
                files.extend([(f, str(f.parent.stem)) for f in sorted(path.glob("shard-*.jsonl.gz"))])
        else:
            raise ValueError(f"Path {path} must be a valid dataset directory")
        
        ds = None
        for file, dataset_name in tqdm(files, desc="Loading files"):
            if file.suffix == ".gz":
                filehandler = gzip.open(file, "rt")
            else:
                filehandler = open(file, "r")
            
            
            with filehandler as f:
                for line in f:
                    data = json.loads(line)
                    if ds is None:
                        ds = {k : [] for k in data.keys()}
                    
                    for k, v in data.items():
                        ds[k].append(v)
        
        ds = Dataset.from_dict(ds)
        ds.push_to_hub(f'nomic-uiuc/stack-mined-{lang.lower()}-v1', private= True)

        

def main(mode, **kwargs):
    eval(mode)(**kwargs)

if __name__ == "__main__":
    fire.Fire(main)


    