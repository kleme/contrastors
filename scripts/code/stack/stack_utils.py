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


VALID_CHARACTERS = set(string.printable)
VALID_CHARACTERS.update(['\t', '\n', '\r', '\f', '\v', '\b', '\0'])
CONTRASTIVE_CODE_DIR = "/tmp/contrastive-code/"



def strip_special_symbols(input_string):
    symbols_to_remove = r'[*#\r\n\t-]'
    cleaned_string = re.sub(symbols_to_remove, '', input_string)
    return cleaned_string

def clean_text(text):
    # Fix bad Unicode text
    text = fix_text(text)
    
    # Remove URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)

    # Remove HTML tags
    html_pattern = re.compile(r'<.*?>')
    text = html_pattern.sub(r'', text)

    # Remove doctags (example: "@doc", "@param", etc.)
    doctag_pattern = re.compile(r'@\w+')
    text = doctag_pattern.sub(r'', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def strip_c_style_comment_delimiters(comment: str) -> str:
    comment_lines = comment.split('\n')
    cleaned_lines = []
    for l in comment_lines:
        l = l.strip()
        if l.endswith('*/'):
            l = l[:-2]
        if l.startswith('*'):
            l = l[1:]
        elif l.startswith('/**'):
            l = l[3:]
        elif l.startswith('//'):
            l = l[2:]
        cleaned_lines.append(l.strip())
    return '\n'.join(cleaned_lines)