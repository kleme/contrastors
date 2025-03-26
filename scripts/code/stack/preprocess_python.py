from datasets import load_dataset
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import string
import os 
from pathlib import Path 
import json
from tqdm import tqdm
import gzip
from torch.utils.data import DataLoader
import re 
from ftfy import fix_text
from stack_utils import VALID_CHARACTERS, CONTRASTIVE_CODE_DIR, strip_special_symbols, clean_text
from preprocess_base import BasePreprocessor


# class PythonProcessor(BasePreprocessor):
#     def extract_docstring(node):
#         for child in node.children:
#             if child.type == 'block' and len(child.children) > 0:
#                 if child.children[0].type == 'expression_statement' and len(child.children[0].children) > 0 and child.children[0].children[0].type == 'string':
#                     exp = child.children[0].children[0].text.decode('utf8')
#                     if '"""' in exp:  #sanity check
#                         return exp
#         return None

#     def process_docstring(query):
#         if any(c not in VALID_CHARACTERS for c in query):
#             return ''
        
#         query = query.split('\n')

#         new_query = []
#         for line in query:
#             if ':' in line:
#                 break 
#             elif any(strip_special_symbols(line).strip() == search for search in ['parameters', 'Parameters', 'InputParameters', 'Input Parameters']):
#                 break

#             new_query.append(line.strip())
        
#         if new_query:
#             new_query = strip_special_symbols(' '.join(new_query)).strip()
#         else:
#             new_query = ''
        
#         if new_query != '':
#             new_query = clean_text(new_query)
            
#         return new_query


#     def get_functions_iterative(root_node, functions, docstrings):
#         stack = [root_node]
#         while stack:
#             node = stack.pop()
#             if node.type == 'function_definition':
#                 docstring = PythonProcessor.extract_docstring(node)
#                 if docstring is not None:
#                     func = node.text.decode('utf8')
#                     docstring_start = func.index(docstring)
#                     functions.append(func[:docstring_start].rstrip().rstrip('\n') + func[(docstring_start + len(docstring)):])
#                     docstrings.append(PythonProcessor.process_docstring(docstring.strip('"""').strip()))

#             stack.extend(reversed(node.children))
            
            
#     def extract_functions_with_docstrings(example, parser):
#         source_code = example['code'][0]

#         tree = parser.parse(bytes(source_code, 'utf8'))
#         root_node = tree.root_node
        
#         functions = []
#         docstrings = []
        
#         PythonProcessor.get_functions_iterative(root_node, functions, docstrings)
#         if functions and docstrings:
#             return {'query': docstrings, 'document': functions}
#         else:
#             return {'query': [''], 'document': ['']}   



parser = Parser(Language(tspython.language()))



        

def extract_docstring(node):
    for child in node.children:
        if child.type == 'block' and len(child.children) > 0:
            if child.children[0].type == 'expression_statement' and len(child.children[0].children) > 0 and child.children[0].children[0].type == 'string':
                exp = child.children[0].children[0].text.decode('utf8')
                if '"""' in exp:  #sanity check
                    return exp
    return None





def process_docstring(query):
    if any(c not in VALID_CHARACTERS for c in query):
        return ''
    
    query = query.split('\n')

    new_query = []
    for line in query:
        if ':' in line:
            break 
        elif any(strip_special_symbols(line).strip() == search for search in ['parameters', 'Parameters', 'InputParameters', 'Input Parameters']):
            break

        new_query.append(line.strip())
    
    if new_query:
        new_query = strip_special_symbols(' '.join(new_query)).strip()
    else:
        new_query = ''
    
    if new_query != '':
        new_query = clean_text(new_query)
        
    return new_query


def get_functions_iterative(root_node, functions, docstrings):
    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.type == 'function_definition':
            docstring = extract_docstring(node)
            if docstring is not None:
                func = node.text.decode('utf8')
                docstring_start = func.index(docstring)
                functions.append(func[:docstring_start].rstrip().rstrip('\n') + func[(docstring_start + len(docstring)):])
                docstrings.append(process_docstring(docstring.strip('"""').strip()))

        stack.extend(reversed(node.children))
        
        
def extract_functions_with_docstrings(example):
    source_code = example['code'][0]

    tree = parser.parse(bytes(source_code, 'utf8'))
    root_node = tree.root_node
    
    functions = []
    docstrings = []
    
    get_functions_iterative(root_node, functions, docstrings)
    if functions and docstrings:
        return {'query': docstrings, 'document': functions}
    else:
        return {'query': [''], 'document': ['']}

# for i in range(10):
#     ds = load_dataset(f"nomic-ai/the-stack-v2-dedup-Python-{i}", split = 'train')
#     ds = ds.filter(lambda x : '"""' in x['code'])
#     ds = ds.map(extract_functions_with_docstrings, batched = True, batch_size= 1, num_proc= 72, remove_columns= ds.column_names)
#     ds = ds.filter(lambda x : (x['query'] is not None and x['document'] is not None and x['query'] != '' and x['document'] != '' and len(x['query'].split()) > 3), num_proc= 72)
#     ds.push_to_hub(f"nomic-uiuc/the-stack-v2-dedup-Python-{i}-processed", private=True)


    
# ds = load_dataset("nomic-ai/the-stack-v2-dedup-Python-0", split = 'train')
# ds = ds.filter(lambda x : '"""' in x['code'])
# ds = ds.map(extract_functions_with_docstrings, batched = True, batch_size= 1, num_proc= 72, remove_columns= ds.column_names)
# ds = ds.filter(lambda x : (x['query'] is not None and x['document'] is not None and x['query'] != '' and x['document'] != '' and len(x['query'].split()) > 3), num_proc= 72)


# #ds.push_to_hub(f"nomic-ai/the-stack-v2-dedup-Python-proc1", private=True)


# dataloader = DataLoader(ds.take(200000), 
#                                     batch_size= 100000, 
#                                     num_workers= 0, 
#                                     persistent_workers= False,
#                                     collate_fn= None, 
#                                     drop_last= False)
            
# output_dir = Path(f"{CONTRASTIVE_CODE_DIR}/stack_v0_chr_ablation/")

# if not output_dir.exists():
#     os.makedirs(output_dir)

# for shard_num, dataset_slice in tqdm(enumerate(dataloader), desc="Writing shards"):
#     with gzip.open(output_dir / f"shard-{shard_num:05d}.jsonl.gz", "wt") as f:
#         for data in tqdm([dict(zip(dataset_slice, v)) for v in zip(*dataset_slice.values())], desc=f"Writing shard {shard_num:05d}"):
#             f.write(json.dumps(data) + "\n")