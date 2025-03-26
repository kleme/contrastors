import nltk
nltk.download('wordnet')
from ast_transformer import perturb
from pathlib import Path
import json
import gzip
from tqdm import tqdm
import argparse
import multiprocessing as mp
import logging 
import concurrent.futures

logging.basicConfig(filename='data_processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

#python scripts/code/augment_data.py --dataset /tmp/contrastive-code/codesearch_full_mined/ --output_dir /tmp/contrastive-code/codesearch_full_mined_random/ --depth 2 --samples 2


TRANSFORMS = ['identity', 'replace_true_false', 'rename', 'add_dead_code', 'unroll_whiles', 'insert_print_statements', 'wrap_try_catch', 'random']

def load_dataset(args):
    path = Path(args.dataset)
    if path.is_dir():
        files = sorted(path.glob("shard-*.jsonl.gz"))
    else:
        files = [path]
    
    dataset = []

    for file in tqdm(files, desc="Loading shards"):
        if file.suffix == ".gz":
            filehandler = gzip.open(file, "rt")
        else:
            filehandler = open(file, "r")
        
        with filehandler as f:
            for line in f:
                data = json.loads(line)
                #data['negatives'] = list(filter(lambda x: x != data['document'], data['negatives']))
                dataset.append(data)
         
    return dataset


def prepare_data(index, data, args, cols):
    prepared = []
    for sample_idx in range(args.samples):
        for col in data.keys():
            if col in cols:
                code = data[col]
                if isinstance(code, str):
                    prepared.append((index, col, code, -1, sample_idx, args))

                elif isinstance(code, list):
                    for j, c in enumerate(code):
                        prepared.append((index, col, c, j, sample_idx, args))
        
    
    return prepared
            

def process_data(item):
    #return format (index, key, col, neg_idx, sample_idx, changed)
    index, key, value, neg_idx, sample_idx, args = item
    if 'def ' not in value:
        return (index, key, value, neg_idx, sample_idx, False)
    result = perturb(value, TRANSFORMS.index(args.transformation), 1, 1, args.psi)
    return (index, key, result[0]['result'], neg_idx, sample_idx, result[0]['changed']) 

def replace_with_transformed(dataset, results, cols):
    added_result = []
    j = 0
    total_processed = len(results)
    
    for i, data in tqdm(enumerate(dataset), total = len(dataset), desc = 'Reassembling Data after Multi-Processing'):
        dct = {k : v for k, v in data.items() if k not in cols}
        while (j < total_processed) and (i == results[j][0]):
            _, k, v, neg_idx, _, _ = results[j]
            if neg_idx == -1:
                dct[k] = v 
            else:
                if k not in dct:
                    dct[k] = []
                    
                dct[k].append(v)
            
            j += 1
        
        added_result.append(dct)
    
    return added_result

def augment_with_transformed(dataset, results, cols):
    added_result = []
    j = 0
    total_processed = len(results)
    
    for i, data in tqdm(enumerate(dataset), total = len(dataset), desc = 'Reassembling Data after Multi-Processing'):
        if (j < total_processed) and (results[j][-1]):
            z = 0
            while (j < total_processed) and (i == results[j][0]):
                
                dct = {k : v for k, v in data.items() if k not in cols}
                
                while (j < total_processed) and (i == results[j][0]) and (results[j][-2] == z):
                    _, k, v, neg_idx, _, _ = results[j]
                    if neg_idx == -1:
                        dct[k] = v 
                    else:
                        if k not in dct:
                            dct[k] = []
                            
                        dct[k].append(v)

                    j += 1
                
                z += 1
                added_result.append(dct)

        
        else:
            while (j < total_processed) and (i == results[j][0]):
                j += 1
            
        

    return added_result
    
        

    
    

def augment_code(dataset, args):
    cols = set(args.cols_to_transform.split())
    
    if not all(isinstance(item, dict) for item in dataset):
        raise ValueError("All items in the dataset must be dictionaries.")
    
    tasks = []
    for i, data in enumerate(dataset):
        tasks.extend(prepare_data(i, data, args, cols))
    
    pool = mp.Pool(mp.cpu_count())
    
    results = []
    
    try:
        for result in tqdm(pool.imap_unordered(process_data, tasks), total=len(tasks), desc='Transforming data'):
            results.append(result)
    except Exception as e:
        pool.terminate()
        raise e
    finally:
        pool.close()
        pool.join()
    
    
    results.sort(key=lambda x: (x[0], x[-2], x[-3]))
    
    if args.mode == 'replace':
        return replace_with_transformed(dataset, results, cols)

    return augment_with_transformed(dataset, results, cols)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", type=str, choices = ['augment', 'replace'], default = "augment")
    parser.add_argument("--transformation", type=str, default = "random", choices = TRANSFORMS)
    parser.add_argument("--cols_to_transform", type=str, default = "document negatives")
    parser.add_argument("--depth", type=int, default = 1)
    parser.add_argument("--samples", type=int, default = 1)
    parser.add_argument("--psi", type=int, default = 1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    dataset = load_dataset(args)

    num_iter = args.samples // 2
    args.samples = 2
    
    
    added_data = []
    for _ in range(num_iter):
        added_data.extend(augment_code(dataset, args))
    
    dataset = added_data
    
    
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents = True, exist_ok= True)
    shard_size = 100_000
    for shard_start in tqdm(range(0, len(dataset), shard_size), desc="Writing shards"):
        dataset_slice = dataset[shard_start : shard_start + shard_size]
        shard_num = shard_start // shard_size
        with gzip.open(output_dir / f"shard-{shard_num:05d}.jsonl.gz", "wt") as f:
            for data in tqdm(dataset_slice, desc=f"Writing shard {shard_num:05d}"):
                f.write(json.dumps(data) + "\n")
    
    
    

