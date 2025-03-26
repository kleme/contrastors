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

import re
import json
import torch
import tokenize
import logging
from io import StringIO
from torch.utils.data import Dataset
import numpy as np
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


class Code2CodeInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
            self,
            code_tokens,
            input_ids,
            index,
            label
    ):
        self.code_tokens = code_tokens
        self.input_ids = input_ids
        self.index = index
        self.label = label


def convert_code2code_examples_to_features(js, tokenizer, args, lang):
    """convert examples to token ids"""
    if "func" in js:
        code = " ".join(remove_comments_and_docstrings(js['func'], lang).split())
    else:
        code = " ".join(remove_comments_and_docstrings(js['code'], lang).split())

    code_tokens = tokenizer.tokenize(code)[:args.code_length - 1]
    code_tokens = code_tokens + [tokenizer.eos_token]
    input_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length

    return Code2CodeInputFeatures(code_tokens, input_ids, js["index"], int(js['label']))


class Code2CodeDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, lang):
        self.examples = []
        data = []
        with open(file_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                data.append(js)

        for js in data:
            self.examples.append(convert_code2code_examples_to_features(js, tokenizer, args, lang))

        self.label_examples = {}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label] = []
            self.label_examples[e.label].append(e)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label))


class NL2CodeInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
            self,
            code_tokens,
            code_ids,
            nl_tokens,
            nl_ids,
            url,
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url


def convert_nl2code_examples_to_features(js, prefix = None):
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code = f'{prefix}: {code}' if prefix is not None else code

    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl = f'{prefix}: {nl}' if prefix is not None else nl

    return code, nl, (js['url'] if "url" in js else js["retrieval_idx"])
    


class NL2CodeDataset:
    def __init__(self, file_path=None, prefix = None):
        self.examples = {'code': [], 'nl': []}
        self.urls = []
        data = []
        self.url2example = {}
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase" in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js)

        for js in data:
            code, nl, url = convert_nl2code_examples_to_features(js, prefix)
            self.examples['code'].append(code)
            self.examples['nl'].append(nl)
            self.urls.append(url)
            self.url2example[url] = [code, nl]
    
    def __len__(self):
        return len(self.urls)

def embed_codesage(model, tokenizer, batch, device, **kwargs):
    encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    inputs = encoded['input_ids']
    attention_mask = inputs.ne(tokenizer.pad_token_id)
    outputs = model(input_ids = inputs.to(device), attention_mask = attention_mask.to(device), return_dict=True)
    return torch.nn.functional.normalize(outputs.pooler_output, p=2, dim=1)

def embed_codet5(model, tokenizer, batch, device, **kwargs):
    encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    return model(**encoded.to(device), return_dict=True)

def embed_unixcoder(model, tokenizer, batch, device, max_length):
    tokens_ids = model.tokenize(batch, max_length= max_length,mode="<encoder-only>", padding = True)
    source_ids = torch.tensor(tokens_ids).to(device)
    return torch.nn.functional.normalize(model(source_ids)[1], p=2, dim=1)

def embed_voyage(model, documents):
    batch_size = 128
    embeddings = []
    toks = 0
    for i in tqdm(range(0, len(documents), batch_size)):
        if toks >= int(1e6) - 1024:
            time.sleep(120)
        embeddings += model.embed(
            documents[i:i + batch_size], model='voyage-code-2', input_type="document"
        ).embeddings
        toks += 128 
    
    return np.array(embeddings)