

import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import smart_open
from tqdm import tqdm
from datasets import load_dataset, Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import Pool
import dill


s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

def fetch_code(blob_id, src_encoding):
    s3_url = f"s3://softwareheritage/content/{blob_id}"
    try:
        with smart_open.open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
            return fin.read().decode(src_encoding)
    except Exception as e:
        return None

def download_the_stack_v2(data_repo="bigcode/the-stack-v2-dedup", language="Java", num_threads=8):
    ds = load_dataset(data_repo, language, split="train", streaming=False)
    
    num_shard = 0
    for i in tqdm(range(0, len(ds), 5000000), colour = 'blue', desc = 'Processing Shards'):
        dataset = {'code': []}

        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(fetch_code, row["blob_id"], row["src_encoding"]) for row in tqdm(ds.select(range(i, min(i + 5000000, len(ds)))))]

            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    dataset['code'].append(result)

        with open(f"cache/the-stack-v2-dedup-{language}-{num_shard}", 'wb') as f:
            dill.dump(dataset, f)
        
        try:
            dataset = Dataset.from_dict(dataset)
            dataset.push_to_hub(f"nomic-ai/the-stack-v2-dedup-{language}-{num_shard}", private=True)
        except:
            pass
        num_shard += 1

download_the_stack_v2(num_threads= os.cpu_count())

# for file in os.listdir('/workspace/contrastors-dev/cache'):
#     with open(f'/workspace/contrastors-dev/cache/{file}', 'rb') as f:
#         dataset = dill.load(f)
    
#     file_name = file.split('/')[-1]
#     dataset = Dataset.from_dict(dataset)
#     dataset.push_to_hub(f"nomic-ai/{file_name}", private=True)








