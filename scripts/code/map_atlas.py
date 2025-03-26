import numpy as np

import gzip
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import nomic
from nomic import atlas
from tqdm import tqdm
from collections import defaultdict
from contrastors.eval.encoder import Encoder, HFEncoder
from contrastors.eval.codesage.utils import embed_codet5
from datasets import load_dataset
from beir.datasets.data_loader import GenericDataLoader

import pandas as pd

NOMIC_TOKEN = os.environ['NOMIC_TOKEN']
HF_ACCESS_TOKEN = os.environ['HF_ACCESS_TOKEN']

nomic.login(NOMIC_TOKEN)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_paths", required=True)
    parser.add_argument("--mode", required=True, choices=['train', 'eval', 'stack', 'coderagbench'])
    parser.add_argument("--atlas_dataset_name", required=True)
    parser.add_argument("--atlas_dataset_description", required=True)
    parser.add_argument("--model_name", default = "")
    parser.add_argument("--tokenizer_name", default = "nomic-ai/nomic-embed-text-v1-unsupervised")
    parser.add_argument("--from_transformers", default = False)
    parser.add_argument("--seq_length", default = 1024)
    parser.add_argument("--device", default = "cuda")
    parser.add_argument("--cols", default = "query document")
    parser.add_argument("--cols_to_melt", default = "retrieved_code top_code")
    parser.add_argument("--embed_col", default = "document")
    parser.add_argument("--prefix", default = None)
    parser.add_argument("--normalize", default = True)
    parser.add_argument("--batch_size", default= 1024)
    return parser.parse_args()



def create_dataset_embeddings(model, dataset, args):
    data_to_embed = [f'{args.prefix}: {sample}' for sample in dataset[args.embed_col].tolist()] if args.prefix is not None else dataset[args.embed_col].tolist()
    
    embeddings = model.encode(data_to_embed, batch_size = args.batch_size)
    mapped_dataset = atlas.map_data(embeddings = np.array(embeddings),
                            data= dataset,
                            identifier= args.atlas_dataset_name,
                            description= args.atlas_dataset_description, 
                            topic_model= dict(topic_label_field = args.embed_col), 
                            is_public= False
                            )
    return mapped_dataset

def load_dataset_train(args):
    columns = args.cols.split()
    dataset = {col: [] for col in [*columns, 'dataset_name']}
    
    col_map = defaultdict(lambda : 'other')
    col_map.update({'pos': 'document',  'document': 'document', 'query': 'query', 'negatives': 'negatives'})
    
    files = []
    for dataset_path in args.dataset_paths.split():
            path = Path(dataset_path)
            if path.exists() and path.is_dir():
                files.extend([(f, str(f.parent.stem)) for f in sorted(path.glob("shard-*.jsonl.gz"))])
            else:
                raise ValueError(f"Path {path} must be a valid dataset directory")
    
    for file, dataset_name in tqdm(files, desc="Loading files"):
        if file.suffix == ".gz":
            filehandler = gzip.open(file, "rt")
        else:
            filehandler = open(file, "r")
        
        with filehandler as f:
            for line in f:
                data = json.loads(line)
                for col in data.keys():
                    mapped_col = col_map[col]
                    if mapped_col in columns:
                        if isinstance(data[col], list):
                            dat = ' '.join(data[col])
                        else:
                            dat = data[col]
                        dataset[mapped_col].append(dat)
                
                dataset['dataset_name'].append(dataset_name)
    
    return pd.DataFrame.from_dict(dataset)

def load_dataset_eval(args):
    df =  pd.read_json(args.dataset_paths, lines=True)[list(args.cols.split())]
    cols_to_melt = args.cols_to_melt.split()
    melted_df = df.melt(id_vars=[col for col in df.columns if col not in cols_to_melt], 
                    value_vars=cols_to_melt,
                    var_name='type', 
                    value_name=args.embed_col)

    melted_df['type'] = melted_df['type'].apply(lambda x: x)
    return melted_df

def load_dataset_stack(args):
    ds = load_dataset(args.dataset_paths, split = 'train').take(1000000)
    return ds.to_pandas()

def load_dataset_coderagbench(args):
    corpus, queries, qrels = GenericDataLoader(data_folder= args.dataset_paths).load(split="test")
    dataset = {'query': [], 'document': []}
    for k, v in qrels.items():
        dataset['query'].append(queries[k])
        dataset['document'].append(corpus[list(v.keys())[0]]['text'])
        
    return pd.DataFrame.from_dict(dataset)
    

if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'train':
        dataset = load_dataset_train(args)
    elif args.mode == 'eval':
        dataset = load_dataset_eval(args)
    elif args.mode == 'coderagbench':
        dataset = load_dataset_coderagbench(args)
    else:
        dataset = load_dataset_stack(args)
    
    import pdb;pdb.set_trace()
    if args.from_transformers:
        model = HFEncoder(args.model_name, seq_length= args.seq_length, embed_fn= embed_codet5)
    else:
        model = Encoder(args.model_name, args.tokenizer_name, seq_length = args.seq_length)
    
    mapped_dataset = create_dataset_embeddings(model, dataset, args)