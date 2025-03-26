from datasets import load_dataset
from tree_sitter import Language, Parser
import tree_sitter_javascript as tsjs
import string
from stack_utils import VALID_CHARACTERS, CONTRASTIVE_CODE_DIR, strip_special_symbols, clean_text, strip_c_style_comment_delimiters
import os 
from pathlib import Path 
import json
from tqdm import tqdm
import gzip
from torch.utils.data import DataLoader
from preprocess_base import BasePreprocessor

#pip install git+https://github.com/tree-sitter/tree-sitter-go

# class JSPreprocessor(BasePreprocessor):
#     def process_docstring(query):
#         if 'Error' in query:
#             return ''
        
#         if '{' in query:
#             return ''
        
#         if any(c not in VALID_CHARACTERS for c in query):
#             return ''
        
#         query = query.split('\n')

#         new_query = []
#         for line in query:
#             if '@' in line:
#                 break 
            
#             new_query.append(line.strip())
        
#         if new_query:
#             new_query = strip_special_symbols(' '.join(new_query)).strip()
#         else:
#             new_query = ''
        
#         if new_query:
#             new_query = clean_text(new_query).strip('/').strip()

#         return new_query


#     def extract_inner_docstring(node):
#         for child in node.children:
#             if child.type == 'block' and len(child.children) >= 2:
#                 comment_buffer = []
#                 for c in child.children[1:]:
#                     if c.type in ('line_comment', 'comment', 'block_comment'):
#                         comment_buffer.append(c.text.decode('utf8'))
#                     else:
#                         break
                
#                 if comment_buffer:
#                     return comment_buffer
                
                
#         return None



#     def get_functions_iterative(root_node, functions, docstrings):
#         stack = [(None, root_node)]
#         while stack:
#             prev, node = stack.pop()
#             if node.type in ('function_definition', 'method_declaration', 'function_declaration', 'constructor_declaration', 'interface_declaration', 'method_definition'):
#                 if prev is not None and prev != []:
#                     docstr = JSPreprocessor.process_docstring('\n'.join([strip_c_style_comment_delimiters(p.text.decode('utf8')) for p in prev]))
#                     #function_name = node.child_by_field_name('name').text.decode('utf8')
#                     if docstr:
#                         docstrings.append(docstr)
#                         functions.append(node.text.decode('utf8'))
#                 else:
#                     inner_docstring = JSPreprocessor.extract_inner_docstring(node)
#                     if inner_docstring:
#                         func = node.text.decode('utf8')
#                         #function_name = node.child_by_field_name('name').text.decode('utf8')
#                         docstring_start = func.index(inner_docstring[0])
#                         docstring_end = func.index(inner_docstring[-1]) + len(inner_docstring[-1])
#                         func = func[:docstring_start].rstrip().rstrip('\n') + func[docstring_end:]
#                         docstr = JSPreprocessor.process_docstring(strip_c_style_comment_delimiters('\n'.join(inner_docstring)))
#                         if docstr:
#                             docstrings.append(docstr)
#                             functions.append(func)
                        
#             children = []
#             comments = []
#             for i in range(len(node.children)):
#                 if i == 0:
#                     children.append((None, node.children[i]))
#                 else:
#                     if node.children[i].type in ('line_comment', 'comment', 'block_comment'):
#                         comments.append(node.children[i])
#                     else:
#                         if node.children[i].type in ('function_definition', 'method_declaration', 'function_declaration', 'constructor_declaration', 'interface_declaration', 'method_definition'):
#                             children.append((comments,node.children[i] ))
                        
#                         else:
#                             children.append((None, node.children[i]))
                            
#                         comments = []

#             stack.extend(reversed(children))



#     def extract_functions_with_docstrings(example, parser):
#         source_code = example['code'][0]
        
#         tree = parser.parse(bytes(source_code, 'utf8'))
#         root_node = tree.root_node
        
        
#         functions = []
#         docstrings = []

#         JSPreprocessor.get_functions_iterative(root_node, functions, docstrings)
        
#         if functions and docstrings:
#             return {'query': docstrings, 'document': functions}
#         else:
#             return {'query': [''], 'document': ['']}  


parser = Parser(Language(tsjs.language()))



def process_docstring(query):
    if 'Error' in query:
        return ''
    
    if '{' in query:
        return ''
    
    if any(c not in VALID_CHARACTERS for c in query):
        return ''
    
    query = query.split('\n')

    new_query = []
    for line in query:
        if '@' in line:
            break 
        
        new_query.append(line.strip())
    
    if new_query:
        new_query = strip_special_symbols(' '.join(new_query)).strip()
    else:
        new_query = ''
    
    if new_query:
        new_query = clean_text(new_query).strip('/').strip()

    return new_query


def extract_inner_docstring(node):
    for child in node.children:
        if child.type == 'block' and len(child.children) >= 2:
            comment_buffer = []
            for c in child.children[1:]:
                if c.type in ('line_comment', 'comment', 'block_comment'):
                    comment_buffer.append(c.text.decode('utf8'))
                else:
                    break
            
            if comment_buffer:
                return comment_buffer
            
            
    return None



def get_functions_iterative(root_node, functions, docstrings):
    stack = [(None, root_node)]
    while stack:
        prev, node = stack.pop()
        if node.type in ('function_definition', 'method_declaration', 'function_declaration', 'constructor_declaration', 'interface_declaration', 'method_definition'):
            if prev is not None and prev != []:
                docstr = process_docstring('\n'.join([strip_c_style_comment_delimiters(p.text.decode('utf8')) for p in prev]))
                #function_name = node.child_by_field_name('name').text.decode('utf8')
                if docstr:
                    docstrings.append(docstr)
                    functions.append(node.text.decode('utf8'))
            else:
                inner_docstring = extract_inner_docstring(node)
                if inner_docstring:
                    func = node.text.decode('utf8')
                    #function_name = node.child_by_field_name('name').text.decode('utf8')
                    docstring_start = func.index(inner_docstring[0])
                    docstring_end = func.index(inner_docstring[-1]) + len(inner_docstring[-1])
                    func = func[:docstring_start].rstrip().rstrip('\n') + func[docstring_end:]
                    docstr = process_docstring(strip_c_style_comment_delimiters('\n'.join(inner_docstring)))
                    if docstr:
                        docstrings.append(docstr)
                        functions.append(func)
                    
        children = []
        comments = []
        for i in range(len(node.children)):
            if i == 0:
                children.append((None, node.children[i]))
            else:
                if node.children[i].type in ('line_comment', 'comment', 'block_comment'):
                    comments.append(node.children[i])
                else:
                    if node.children[i].type in ('function_definition', 'method_declaration', 'function_declaration', 'constructor_declaration', 'interface_declaration', 'method_definition'):
                        children.append((comments,node.children[i] ))
                    
                    else:
                        children.append((None, node.children[i]))
                        
                    comments = []

        stack.extend(reversed(children))



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

# for i in range(16, 18):
#     ds = load_dataset(f"nomic-ai/the-stack-v2-dedup-JavaScript-{i}", split='train')
#     ds = ds.map(extract_functions_with_docstrings, batched=True, batch_size=1, num_proc=72, remove_columns=ds.column_names)
#     ds = ds.filter(lambda x: (x['query'] is not None and x['document'] is not None and x['query'] != '' and x['document'] != '' and len(x['query'].split()) > 3), num_proc=72)
#     ds.push_to_hub(f"nomic-uiuc/the-stack-v2-dedup-JavaScript-{i}-processed", private=True)


# shard_num = 94
# for i in range(15, 18):    
#     ds = load_dataset(f"nomic-ai/the-stack-v2-dedup-JavaScript-{i}", split = 'train')
#     #ds = ds.filter(lambda example, idx: idx > 500000, with_indices=True, num_proc = 72)
#     ds = ds.map(extract_functions_with_docstrings, batched = True, batch_size= 1,  num_proc= 72, remove_columns= ds.column_names)
#     ds = ds.filter(lambda x : (x['query'] is not None and x['document'] is not None and x['query'] != '' and x['document'] != '' and len(x['query'].split()) > 3), num_proc= 72)



#     dataloader = DataLoader(ds.shuffle(seed = 42), 
#                                         batch_size= 100000, 
#                                         num_workers= 0, 
#                                         persistent_workers= False,
#                                         collate_fn= None, 
#                                         drop_last= False)
                
#     output_dir = Path(f"{CONTRASTIVE_CODE_DIR}/stack_v0_most/javascript")

#     if not output_dir.exists():
#         os.makedirs(output_dir)

#     for _, dataset_slice in tqdm(enumerate(dataloader), desc="Writing shards"):
#         with gzip.open(output_dir / f"shard-{shard_num:05d}.jsonl.gz", "wt") as f:
#             for data in tqdm([dict(zip(dataset_slice, v)) for v in zip(*dataset_slice.values())], desc=f"Writing shard {shard_num:05d}"):
#                 f.write(json.dumps(data) + "\n")
        
#         shard_num += 1