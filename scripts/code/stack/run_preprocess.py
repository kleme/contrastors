from preprocess_go import GoPreprocessor
from preprocess_java import JavaPreprocessor
from preprocess_js import JSPreprocessor
from preprocess_php import PHPProcessor 
from preprocess_python import PythonProcessor
from preprocess_ruby import RubyProcessor 
import tree_sitter_python as tspython
import tree_sitter_go as tsgo 
import tree_sitter_java as tsjava 
import tree_sitter_javascript as tsjs 
import tree_sitter_php as tsphp 
import tree_sitter_ruby as tsruby
from tree_sitter import Language, Parser
from datasets import load_dataset 
from torch.utils.data import DataLoader
import os
from tqdm import tqdm 
import gzip 
import fire
import json 
from pathlib import Path

def main(language, hf_org, max_key, upload_mode, output_dir):
    
    if language == 'Python':
        preprocessor = PythonProcessor
        parser = Parser(Language(tspython.language()))
    
    elif language == 'Go':
        preprocessor = GoPreprocessor 
        parser = Parser(Language(tsgo.language())) 
    
    elif language == 'Java':
        preprocessor = JavaPreprocessor
        parser = Parser(Language(tsjava.language()))
    
    elif language == 'JavaScript':
        preprocessor = JSPreprocessor
        parser = Parser(Language(tsjs.language()))
    
    elif language == 'Ruby':
        preprocessor = RubyProcessor
        parser = Parser(Language(tsruby.language()))
    
    elif language == 'PHP':
        preprocessor = PHPProcessor
        parser = Parser(Language(tsphp.language_php()))
    
    else:
        raise NotImplementedError('language not supported')
    
    max_key = int(max_key)
    
    for i in range(max_key):
        ds = load_dataset(f"nomic-ai/the-stack-v2-dedup-{language}-{i}", split = 'train')
        if language == 'Python':
            ds = ds.filter(lambda x : '"""' in x['code'])
        
        ds = ds.map(preprocessor.extract_functions_with_docstrings, batched = True, batch_size= 1, num_proc= os.cpu_count(), remove_columns= ds.column_names, fn_kwargs= {'parser': parser})
        ds = ds.filter(lambda x : (x['query'] is not None and x['document'] is not None and x['query'] != '' and x['document'] != '' and len(x['query'].split()) > 3), num_proc= os.cpu_count())
        
        if upload_mode == 'hf':
            ds.push_to_hub(f"{hf_org}/the-stack-v2-dedup-{language}-{i}-processed", private=True)
        
        elif upload_mode == 'jsonl':
            dataloader = DataLoader(ds, 
                                    batch_size= 100000, 
                                    num_workers= 0, 
                                    persistent_workers= False,
                                    collate_fn= None, 
                                    drop_last= False)
            
            output_dir = Path(f"{output_dir}/{language}/")

            if not output_dir.exists():
                os.makedirs(output_dir)

            for shard_num, dataset_slice in tqdm(enumerate(dataloader), desc="Writing shards"):
                with gzip.open(output_dir / f"shard-{shard_num:05d}.jsonl.gz", "wt") as f:
                    for data in tqdm([dict(zip(dataset_slice, v)) for v in zip(*dataset_slice.values())], desc=f"Writing shard {shard_num:05d}"):
                        f.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    fire.Fire(main)