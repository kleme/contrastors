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

#from contrastors.models.encoder import BertConfig, NomicBertModel, bert_config_to_nomic_config

#CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 scripts/text/filter_mine.py --dataset=/home/tarun/contrastors-code/src/contrastors/data/stack_v0_prefilter/python --output_dir=src/contrastors/data/python_full_mined1/ --query_key="query" --document_key='document' --negatives_key='negatives'

#torchrun --nproc-per-node=2 scripts/text/filter_mine.py --dataset=src/contrastors/data/stack_v0_prefilter/ruby --output_dir=src/contrastors/data/ruby_full_mined1/ --query_key="query" --document_key='document' --negatives_key='negatives'

#CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 scripts/text/filter_mine.py --dataset=/home/tarun/contrastors-code/src/contrastors/data/stack_v0_prefilter/ruby --output_dir=src/contrastors/data/ruby_full_mined1/ --query_key="query" --document_key='document' --negatives_key='negatives' --max_files 10
#torchrun --nproc-per-node=2 scripts/text/filter_mine.py --dataset=/home/tarun/contrastors-code/src/contrastors/data/stack_v0_prefilter/python --output_dir=src/contrastors/data/python_full_mined1/ --query_key="query" --document_key='document' --negatives_key='negatives' --max_files 100

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--query_key", required=True)
    parser.add_argument("--document_key", required=True)
    parser.add_argument("--negatives_key", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--index_size", type=int, default=1_000_000)
    parser.add_argument("--k", type=int, default=101)
    parser.add_argument("--max_files", type=int, default=1000)
    parser.add_argument("--file_start", type=int, default=0)

    return parser.parse_args()


def send_dict_to_rank0(tensor_dict):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Separate keys and tensors, ensuring that the keys are sorted to maintain order
    keys = sorted(tensor_dict.keys())
    tensors = [tensor_dict[k] for k in keys]
    queries = [tensor[0] for tensor in tensors]
    documents = [tensor[1] for tensor in tensors]

    # On rank 0, prepare lists to gather tensors
    if rank == 0:
        gathered_queries = [[] for _ in range(len(queries))]
        gathered_documents = [[] for _ in range(len(documents))]
    else:
        gathered_queries = None
        gathered_documents = None

    # Gather tensors on rank 0
    for i, (query, document) in enumerate(zip(queries, documents)):
        # On rank 0, prepare a list to store gathered tensors from all ranks for the current tensor
        if rank == 0:
            gathered_q = [torch.empty_like(query) for _ in range(world_size)]
            gathered_queries[i] = gathered_q

            gathered_d = [torch.empty_like(document) for _ in range(world_size)]
            gathered_documents[i] = gathered_d

        else:
            gathered_q = None
            gathered_d = None

        # Gather the current tensor across all ranks
        dist.gather(query, gather_list=gathered_q, dst=0)
        dist.gather(document, gather_list=gathered_d, dst=0)

    if rank == 0:
        gathered_keys = [[] for _ in range(world_size)]
    else:
        gathered_keys = None

    dist.gather_object(keys, object_gather_list=gathered_keys, dst=0)

    if rank == 0:
        # Flatten the lists of keys and tensors
        flat_keys = [item for sublist in gathered_keys for item in sublist]
        flat_queries = [item for sublist in zip(*gathered_queries) for item in sublist]
        flat_documents = [item for sublist in zip(*gathered_documents) for item in sublist]

        # Rebuild the dictionary
        rebuilt_dict = {k: (q, d) for k, q, d in zip(flat_keys, flat_queries, flat_documents)}
        return rebuilt_dict
    else:
        return None


def print_rank0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def load_dataset(path, query_key, document_key, file_start=0, max_files=100):
    dataset = []
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("shard-*.jsonl.gz"))[file_start : file_start + max_files]
    else:
        files = [path]

    i = 0
    for file in tqdm(files, desc="Loading shards", disable=dist.get_rank() != 0):
        filehandler = gzip.open(file, "rt") if file.suffix == ".gz" else open(file, "r")
        with filehandler as f:
            for line in f:
                data = json.loads(line)
                record = {query_key: data[query_key], document_key: data[document_key], "id": i}

                dataset.append(record)
                i += 1

    return dataset


def get_num_lines(dataset):
    num_lines = 0
    total_bytes = os.path.getsize(dataset)
    progbar = tqdm(total=total_bytes, unit="B", unit_scale=True, disable=dist.get_rank() != 0)
    if dataset.is_dir():
        files = sorted(dataset.glob("shard-*.jsonl.gz"))
    else:
        files = [dataset]
    for file in tqdm(files):
        filehandler = gzip.open(file, "rt") if file.endswith(".gz") else open(file, "r")
        with filehandler as f:
            for _ in f:
                num_lines += 1
                progbar.update(f.buffer.fileobj.tell() - progbar.n)

    return num_lines


def dict_collator(records, tokenizer, query_key, document_key, per_device_batch_size):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    batch = {"query": [], "document": [], "id": []}

    for i, record in enumerate(records):
        if i % world_size != rank:
            continue

        batch["query"].append(record[query_key])
        batch["document"].append(record[document_key])
        batch["id"].append(record["id"])

        if len(batch["query"]) == per_device_batch_size:
            tokenized_query = tokenizer(batch["query"], padding=True, truncation=True, return_tensors="pt")
            tokenized_document = tokenizer(batch["document"], padding=True, truncation=True, return_tensors="pt")
            yield {"query": tokenized_query, "document": tokenized_document, "id": batch["id"]}
            batch = {"query": [], "document": [], "id": []}

    # if we have a partial batch, yield it
    if len(batch["query"]) > 0:
        tokenized_query = tokenizer(batch["query"], padding=True, truncation=True, return_tensors="pt")
        tokenized_document = tokenizer(batch["document"], padding=True, truncation=True, return_tensors="pt")
        yield {"query": tokenized_query, "document": tokenized_document, "id": batch["id"]}


def jsonl_collator(path, tokenizer, query_key, document_key, per_device_batch_size):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    batch = {"query": [], "document": [], "id": []}

    filehandler = gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")
    with filehandler as f:
        for i, line in enumerate(f):
            if i % world_size != rank:
                continue

            data = json.loads(line)
            if isinstance(data, list):
                batch["query"].append(data[0])
                batch["document"].append(data[1])
                batch["id"].append(i)
            else:
                batch["query"].append(data[query_key])
                if isinstance(data[document_key], list):
                    # take first since it's easy to do and we don't have to find before
                    batch["document"].append(data[document_key][0])
                else:
                    batch["document"].append(data[document_key])

                batch["id"].append(i)

            if len(batch["query"]) == per_device_batch_size:
                tokenized_query = tokenizer(batch["query"], padding=True, truncation=True, return_tensors="pt")
                tokenized_document = tokenizer(batch["document"], padding=True, truncation=True, return_tensors="pt")
                yield {"query": tokenized_query, "document": tokenized_document, "id": batch["id"]}
                batch = {"query": [], "document": [], "id": []}

        # if we have a partial batch, yield it
        if len(batch["query"]) > 0:
            tokenized_query = tokenizer(batch["query"], padding=True, truncation=True, return_tensors="pt")
            tokenized_document = tokenizer(batch["document"], padding=True, truncation=True, return_tensors="pt")
            yield {"query": tokenized_query, "document": tokenized_document, "id": batch["id"]}


def embed(model, dataloader, batch_size, max_samples):
    id2embedding = {}
    examples_seen = 0
    progbar = tqdm(total=max_samples // batch_size + 1, disable=dist.get_rank() != 0)
    with torch.no_grad():
        for batch in dataloader:
            ids = batch.pop("id")
            query_inputs = {k: v.to(f"cuda:{dist.get_rank()}") for k, v in batch["query"].items()}
            query = model(**query_inputs)

            query = mean_pooling(query, query_inputs["attention_mask"])
            normalized_query = F.normalize(query, p=2, dim=1)

            answer_inputs = {k: v.to(f"cuda:{dist.get_rank()}") for k, v in batch["document"].items()}
            answer = model(**answer_inputs)

            answer = mean_pooling(answer, answer_inputs["attention_mask"])
            normlized_answer = F.normalize(answer, p=2, dim=1)

            id2embedding.update(
                {id: (query.cpu(), answer.cpu()) for id, query, answer in zip(ids, normalized_query, normlized_answer)}
            )

            progbar.update(1)
            examples_seen += batch_size
            if examples_seen >= max_samples:
                break

    return id2embedding

def analyze_points(id2embeddings, batch_size=256, top_k=10):
    index = faiss.IndexFlatIP(len(id2embeddings[list(id2embeddings.keys())[0]][0]))
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = True
    print("building index")
    index = faiss.index_cpu_to_all_gpus(index, co=co)
    print("index built")
    
    # Separate document and query embeddings
    id2doc_emb = {k: v[1] for k, v in id2embeddings.items()}
    id2query_emb = {k: v[0] for k, v in id2embeddings.items()}
    range2id = {i: id for i, id in enumerate(sorted(id2doc_emb.keys()))}
    doc_emb = [id2doc_emb[range2id[i]] for i in range(len(range2id))]

    # Add document embeddings to FAISS index
    index.add(np.array(doc_emb).astype(np.float32))

    # Initialize the results dictionary
    results = {}

    # Process the embeddings in batches
    for i in tqdm(range(0, len(range2id), batch_size), disable=dist.get_rank() != 0):
        atlas_ids = [range2id[j] for j in range(i, min(i + batch_size, len(range2id)))]
        query_embs = [id2query_emb[atlas_id] for atlas_id in atlas_ids]
        
        # Perform search with FAISS
        scores, top_k_indices = index.search(np.array(query_embs).astype(np.float32), top_k)
        
        for j, atlas_id in enumerate(atlas_ids):
            top_ids = [range2id[idx] for idx in top_k_indices[j]]
            top_scores = scores[j]
            
            # Check if the ground truth document is in the top-k
            if atlas_id in top_ids:
                doc_rank = top_ids.index(atlas_id)
                doc_score = top_scores[doc_rank]
            else:
                doc_rank = -1
                doc_score = 0.0
            
            # Extract negative IDs and scores
            negatives = [top_ids[k] for k in range(top_k) if top_ids[k] != atlas_id]
            negative_scores = [top_scores[k] for k in range(top_k) if top_ids[k] != atlas_id]
            
            # Store the results
            results[atlas_id] = {
                'document_score': doc_score,
                'document_rank': doc_rank,
                'negatives': negatives,
                'negative_scores': negative_scores
            }

    return results

def filter_points(id2embeddings, batch_size=256):
    index = faiss.IndexFlatIP(len(id2embeddings[list(id2embeddings.keys())[0]][0]))
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = True
    print("building index")
    index = faiss.index_cpu_to_all_gpus(index, co=co)
    print("index built")

    id2doc_emb = {k: v[1] for k, v in id2embeddings.items()}
    id2_query_emb = {k: v[0] for k, v in id2embeddings.items()}
    range2id = {i: id for i, id in enumerate(sorted(id2doc_emb.keys()))}
    doc_emb = [id2doc_emb[range2id[i]] for i in range(len(range2id))]

    index.add(np.array(doc_emb).astype(np.float32))

    ids2keep = []
    for i in tqdm(range(0, len(range2id), batch_size), disable=dist.get_rank() != 0):
        atlas_ids = [range2id[j] for j in range(i, min(i + batch_size, len(range2id)))]
        query_embs = [id2_query_emb[atlas_id] for atlas_id in atlas_ids]
        _, top_k_indices = index.search(np.array(query_embs).astype(np.float32), 2)
        valid_pairs = (
            np.equal(top_k_indices, np.arange(i, min(i + batch_size, len(range2id)))[:, None]).sum(axis=1).tolist()
        )
        for j, is_valid in enumerate(valid_pairs):
            if is_valid:
                ids2keep.append(atlas_ids[j])

    return ids2keep

def filter_dataset(args, dataset, output_dir, mined_dct):
    metadata = {
        "objective": {"self": [], "paired": [], "triplet": [[args.query_key, args.document_key, args.negatives_key]]}
    }
    path = Path(dataset)
    if path.is_dir():
        files = sorted(path.glob("shard-*.jsonl.gz"))[args.file_start : args.file_start + args.max_files]
    else:
        files = [path]
    
    dataset = []
    i = 0
    for file in tqdm(files, desc="Loading shards"):
        if file.suffix == ".gz":
            filehandler = gzip.open(file, "rt")
        else:
            filehandler = open(file, "r")
        
        with filehandler as f:
            for line in f:
                data = json.loads(line)
                dataset.append(data)

    new_dataset = []
    for i, data in enumerate(dataset):
        if i in mined_dct:
            mined_res = mined_dct[i]
            data['metadata'] = metadata
            
            filtered_negs = []
            filtered_neg_scores = []
            
            for x, scr in zip(mined_res['negatives'], mined_res['negative_scores']):
                neg = dataset[x]['document']
                if neg != data['document']:
                    filtered_negs.append(neg)
                    filtered_neg_scores.append(scr)
                else:
                    data['document_score'] = scr
            
            data['negatives'] = filtered_negs
            data['negative_scores'] = [str(scr) for scr in filtered_neg_scores]
            data['document_score'] = str(mined_res.get('document_score', 0.0))
            data['document_rank'] = str(mined_res.get('document_rank', -1))
            
            new_dataset.append(data)
    
    dataset = new_dataset
    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok= True)
    shard_size = 100_000
    shard_num = args.file_start
    for shard_start in tqdm(range(0, len(dataset), shard_size), desc="Writing shards"):
        dataset_slice = dataset[shard_start : shard_start + shard_size]
        with gzip.open(output_dir / f"shard-{shard_num:05d}.jsonl.gz", "wt") as f:
            for data in tqdm(dataset_slice, desc=f"Writing shard {shard_num:05d}"):
                f.write(json.dumps(data) + "\n")
        shard_num += 1


if __name__ == "__main__":
    dist.init_process_group(timeout=timedelta(minutes=60))
    torch.cuda.set_device(dist.get_rank())
    args = parse_args()

    output_dir = Path(args.output_dir)
    if dist.get_rank() == 0:
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    dataset = Path(args.dataset)
    if dataset.is_dir():
        records = load_dataset(dataset, args.query_key, args.document_key, args.file_start, args.max_files)
        num_lines = len(records)
    else:
        num_lines = get_num_lines(dataset)

    print_rank0(f"num lines: {num_lines}")

    num_examples_per_rank = math.ceil(num_lines / dist.get_world_size())
    print_rank0(f"num examples per rank: {num_examples_per_rank}")
    num_iterations = num_lines // args.index_size
    print_rank0(f"num iterations: {num_iterations}")

    per_device_max_samples = min(args.index_size // dist.get_world_size(), num_examples_per_rank)
    print_rank0(f"Total examples per device: {per_device_max_samples}")
    num_batches_per_device = per_device_max_samples // args.batch_size
    if per_device_max_samples % args.batch_size != 0:
        num_batches_per_device += 1
    print_rank0(f"Num batchers per device: {num_batches_per_device}")

    model_name = 'jinaai/jina-embeddings-v2-base-code'
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(f"cuda:{dist.get_rank()}").to(torch.float16)


    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512

    # initialize once in case we have more than one iteration
    if dataset.is_dir():
        dataloader = dict_collator(records, tokenizer, args.query_key, args.document_key, args.batch_size)
    else:
        dataloader = jsonl_collator(dataset, tokenizer, args.query_key, args.document_key, args.batch_size)

    total_samples = 0
    total_kept = 0
    res_dct = {}
    if num_iterations == 0:
        num_iterations = 1
    for i in tqdm(range(num_iterations), disable=dist.get_rank() != 0):
        # if we're on the last iteration and it's not divisible by batch_size * world_size, round down
        if i == num_iterations - 1:
            total_seen = i * num_batches_per_device * args.batch_size * dist.get_world_size()
            remaining = num_lines - total_seen
            per_device_max_samples = remaining - (remaining % (args.batch_size * dist.get_world_size()))
            per_device_max_samples = (per_device_max_samples // dist.get_world_size()) - 1

        print(f"rank {dist.get_rank()} embedding {per_device_max_samples} samples")
        embeddings = embed(model, dataloader, args.batch_size, per_device_max_samples)
        print(f"rank {dist.get_rank()} finished embedding {len(embeddings)} samples")

        dist.barrier()
        all_embeddings = send_dict_to_rank0(embeddings)

        if dist.get_rank() == 0:
            torch.cuda.empty_cache()
            all_embeddings = {k: (q.numpy(), d.numpy()) for k, (q, d) in all_embeddings.items()}
            res_dct.update(analyze_points(all_embeddings,top_k= args.k))

        dist.barrier()
    filter_dataset(args, args.dataset, args.output_dir, res_dct)