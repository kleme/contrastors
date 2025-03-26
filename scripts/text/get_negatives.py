import gzip
import json
import math
import os
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

#CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 scripts/text/get_negatives.py --dataset=src/contrastors/data/cosqa_full_mined/ --output_dir=contrastors-code/src/contrastors/data/cosqa_full_mined2/ --query_key="query" --document_key='document' --negatives_key='negatives'

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--k", type=int, default=101)
    parser.add_argument("--query_key", default="question")
    parser.add_argument("--document_key", default="positive_ctxs")
    parser.add_argument("--negatives_key", default="hard_negative_ctxs")
    parser.add_argument("--add_title", action="store_true")

    return parser.parse_args()


def load_dataset(path, query_key, document_key, negatives_key):
    queries = []
    documents = []
    all_data = []
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("shard-*.jsonl.gz"))
    else:
        files = [path]
    seen_documents = set()
    for file in tqdm(files, desc="Loading shards"):
        if file.suffix == ".gz":
            filehandler = gzip.open(file, "rt")
        else:
            filehandler = open(file, "r")
        with filehandler as f:
            for line in f:
                data = json.loads(line)
                queries.append({query_key: data[query_key]})
                docs = data[document_key]

                if docs not in seen_documents:
                    documents.append({document_key: docs})

                if negatives_key in data:
                    data.pop(negatives_key)
                    # negatives = data[negatives_key]

                    # # nq format is in list for whatever reason
                    # if isinstance(negatives, str):
                    #     negatives = [{document_key: negatives}]
                    # elif isinstance(negatives, list):
                    #     if isinstance(negatives[0], str):
                    #         negatives = [{document_key: neg} for neg in negatives]
                    # else:
                    #     raise ValueError(f"Unknown format for negatives: {negatives}")

                    # seen_documents.update([neg[document_key] for neg in negatives])

                    # documents.extend([neg for neg in negatives if neg[document_key] not in seen_documents])

                seen_documents.add(docs)

                all_data.append(data)

    return queries, documents, all_data


def print_rank0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed(model, tokenizer, dataset, batch_size, key, add_title=False):
    embeddings = []
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(dataset), batch_size), desc=f"Embedding {key}"):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch = dataset[batch_start:batch_end]
            if add_title:
                batch = [line["title"] + " " + line[key] for line in batch]
            else:
                batch = [line[key] for line in batch]

            tokenized = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)

            model_output = model(**tokenized)
            pooled = mean_pooling(model_output, tokenized["attention_mask"])
            normalized_emb = F.normalize(pooled, p=2, dim=1)
            embeddings.extend(normalized_emb.detach().cpu().numpy())

    return embeddings


def knn_neighbors(queries, index, batch_size, k):
    all_scores, all_indices = [], []
    for i in tqdm(range(0, len(queries), batch_size), disable=dist.get_rank() != 0):
        query_embs = queries[i : i + batch_size]
        top_k_scores, top_k_indices = index.search(np.array(query_embs).astype(np.float32), k)

        all_scores.extend(top_k_scores)
        all_indices.extend(top_k_indices)

    return all_scores, all_indices


if __name__ == "__main__":
    dist.init_process_group(timeout=timedelta(minutes=60))
    torch.cuda.set_device(dist.get_rank())
    args = parse_args()

    output_dir = Path(args.output_dir)
    if dist.get_rank() == 0:
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    model_name = 'jinaai/jina-embeddings-v2-base-code'
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code = True).to(f"cuda:{dist.get_rank()}")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512

    # initialize once in case we have more than one iteration
    queries, documents, dataset = load_dataset(args.dataset, args.query_key, args.document_key, args.negatives_key)

    q_embed = embed(model, tokenizer, queries, args.batch_size, args.query_key)
    d_embed = embed(model, tokenizer, documents, args.batch_size, args.document_key, add_title=args.add_title)

    del model
    torch.cuda.empty_cache()

    index = faiss.IndexFlatIP(len(q_embed[0]))
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = True
    index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(np.array(d_embed).astype(np.float32))
    scores, indices = knn_neighbors(q_embed, index, args.batch_size, args.k)
    for i, data in enumerate(tqdm(dataset)):
        query = data[args.query_key]
        inxs = indices[i]
        scrs = scores[i]
        filtered_inx = []
        filtered_scr = []
        for inx, scr in zip(inxs, scrs):
            if inx == -1:
                break
            selected_doc = documents[inx][args.document_key]
            # assuming this is NQ
            if isinstance(data[args.document_key], list):
                raise NotImplementedError('not tested')
                
            else:
                if selected_doc != data[args.document_key] and selected_doc != query:
                    filtered_inx.append(inx)
                    filtered_scr.append(scr)
                elif selected_doc == data[args.document_key]:
                    data['document_score'] = scr

        data[args.negatives_key] = [documents[inx][args.document_key] for inx in filtered_inx]
        data['negative_scores'] = filtered_scr
        if len(data[args.negatives_key]) < args.k:
            remaining = args.k - len(data[args.negatives_key])
            while True:
                random_idx = np.random.randint(0, len(documents), size=remaining).tolist()
                kept_idxs = []
                kept_scrs = []
                for random in random_idx:
                    if random == -1:
                        continue
                    if isinstance(data[args.document_key], list):
                        raise NotImplementedError
                    else:
                        if documents[random] != data[args.document_key] and documents[random] != query:
                            kept_idxs.append(documents[random][args.document_key])
                            scr = 0.0
                            for doc, nscor in zip(data[args.negatives_key], data['negative_scores']):
                                if documents[random] == doc:
                                    scr = nscor
                                    break
                            kept_scrs.append(nscor)
                        elif  documents[random] == data[args.document_key]:
                            data['document_score'] = 0.0
                if len(kept_idxs) == remaining:
                    break

            data["negatives"].extend(kept_idxs)
            data['negative_scores'].extend(kept_scrs)

    metadata = {
        "objective": {"self": [], "paired": [], "triplet": [[args.query_key, args.document_key, args.negatives_key]]}
    }
    shard_size = 100_000
    for shard_start in tqdm(range(0, len(dataset), shard_size), desc="Writing shards"):
        dataset_slice = dataset[shard_start : shard_start + shard_size]
        for record in dataset_slice:
            record["metadata"] = metadata
            filtered_negs = []
            filtered_neg_scores = []
            for x, scr in zip(record['negatives'], record['negative_scores']):
                if x != record['document']:
                    filtered_negs.append(x)
                    filtered_neg_scores.append(scr)
                else:
                    record['document_score'] = scr     
            
            record['negatives'] = filtered_negs
            record['negative_scores'] = [str(scr) for scr in filtered_neg_scores]
            record['document_score'] = str(record.get('document_score', 0.0))
        shard_num = shard_start // shard_size
        with gzip.open(output_dir / f"shard-{shard_num:05d}.jsonl.gz", "wt") as f:
            for data in tqdm(dataset_slice, desc=f"Writing shard {shard_num:05d}"):
                f.write(json.dumps(data) + "\n")
