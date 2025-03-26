# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import random
import logging
import argparse
from pathlib import Path
import json
import numpy as np
import shutil

from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from contrastors.eval.codesage.utils import NL2CodeDataset, embed_codesage, embed_codet5, embed_unixcoder, embed_voyage
from transformers import AutoTokenizer, AutoConfig, AutoModel
from contrastors.models.biencoder import BiEncoder, BiEncoderConfig
from contrastors.eval.encoder import Encoder, HFEncoder
from functools import partial
from contrastors.eval.codesage.unixcoder import UniXcoder
import voyageai
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_model(args, max_length):
    if args.model_type == 'transformers':
        if 't5' in args.model_name_or_path:
            model = HFEncoder(args.model_name_or_path, max_length, embed_fn= embed_codet5)
        elif 'codesage' in args.model_name_or_path:
            model = HFEncoder(args.model_name_or_path, max_length, embed_fn= embed_codesage)
        elif 'unixcoder' in args.model_name_or_path:
            model = UniXcoder(args.model_name_or_path)
            model = HFEncoder.from_other(model, None, max_length, partial(embed_unixcoder, max_length = max_length))
            model.device = 'cuda'
        else:
            model = HFEncoder(args.model_name_or_path, max_length)
    else:
        model = Encoder(args.model_name_or_path, args.tokenizer_name_or_path, max_length, is_biencoder= True)
    
    return model
    

def evaluate(args):
    prefixes = None
    if args.prefixes is not None and args.prefixes != '':
        prefixes = args.prefixes.split()
        assert len(prefixes) == 2, "Invalid prefixes length"
    
    if args.model_name_or_path == 'voyage':
        model = voyageai.Client(os.environ.get("VOYAGE_API_KEY"))
    else:
        model = get_model(args, args.nl_length)
        
    query_dataset = NL2CodeDataset(args.test_data_file, prefix = prefixes[0] if prefixes is not None else None)


    code_dataset = NL2CodeDataset(args.codebase_file, prefix = prefixes[1] if prefixes is not None else None)

    logger.info("***** Running evaluation *****")
    logger.info("Num queries = %d", len(query_dataset))
    logger.info("Num codes = %d", len(code_dataset))
    logger.info("Batch size = %d", args.eval_batch_size)
    
    if args.model_name_or_path == 'voyage':
        nl_vecs = embed_voyage(model, query_dataset.examples['nl'])
        code_vecs =  embed_voyage(model, code_dataset.examples['code'])
        import pdb;pdb.set_trace()
    else:
        nl_vecs = model.encode(query_dataset.examples['nl'], batch_size = args.eval_batch_size)
        del model 
        model = get_model(args, args.code_length)
        code_vecs = model.encode(code_dataset.examples['code'], batch_size = args.eval_batch_size)

        code_vecs = np.stack(code_vecs, axis = 0)
        nl_vecs = np.stack(nl_vecs, axis = 0)

    scores = np.matmul(nl_vecs, code_vecs.T)
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
    print(f"nl_vecs_shape: {nl_vecs.shape} "
          f"\t code_vecs_shape: {code_vecs.shape} "
          f"\t score_matrix_shape: {scores.shape}")

    nl_urls = []
    code_urls = []
    for url in query_dataset.urls:
        nl_urls.append(url)

    for url in code_dataset.urls:
        code_urls.append(url)

    ranks = []
    result = []
    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False
        for i, idx in enumerate(sort_id[:1000]):
            if i == 0:
                result.append({'query': query_dataset.url2example[url][1], 'retrieved_code': code_dataset.url2example[url][0], 'top_code': code_dataset.url2example[code_urls[idx]][0]})
                
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            result[-1]['rank'] = rank
            ranks.append(1 / rank)
        else:
            result[-1]['rank'] = None
            ranks.append(0)

    mrr =  float(np.mean(ranks))
    
    return mrr, result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_name", default=None, type=str,
                        help="the name of the nl2code eval set")
    parser.add_argument("--language", default="python", type=str,
                        help="the name of the nl2code eval set")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--model_type", choices = ['transformers', 'nomic'], default = 'transformers', type=str,
                        help="The model type")
    parser.add_argument("--prefixes", type = str, default = None,
                        help="prefixes")
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--result_dir", required=True, type=str, help="path to store the evaluation results.")

    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args.seed)
    logger.info("device: %s, n_gpu: %s, seed: %s", device, args.n_gpu, args.seed)

    mrr, result = evaluate(args)
    logger.info("***** Eval results *****")
    logger.info("EVAL_MRR = %s", str(round(mrr * 100, 2)))

    model_dir_name = args.model_name_or_path.split("/")[-2] if args.model_type == 'nomic' else args.model_name_or_path.split("/")[-1]

    result_dir = f"{args.result_dir}/{model_dir_name}"
    
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    result_data = {
        "language": args.language,
        "model_dir_name": model_dir_name,
    }
    
    with open(f"{result_dir}/overall_results.jsonl", 'a') as f:
        f.write(json.dumps({**result_data, **{'mrr': mrr}}) + "\n")
    
    with open(f"{result_dir}/result_query_code.jsonl", 'a') as f:
        for res in result:
            f.write(json.dumps({**result_data, **res}) + "\n")
            
        
    
    

if __name__ == "__main__":
    main()