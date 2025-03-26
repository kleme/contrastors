import tree_sitter_python as tspython
from tree_sitter import Language, Parser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NODE_TYPES = ['!=',
 '%',
 '%=',
 '&',
 '&=',
 '(',
 ')',
 '*',
 '**',
 '**=',
 '*=',
 '+',
 '+=',
 ',',
 '-',
 '-=',
 '->',
 '.',
 '/',
 '//',
 '//=',
 '/=',
 ':',
 ':=',
 ';',
 '<',
 '<<',
 '<<=',
 '<=',
 '<>',
 '=',
 '==',
 '>',
 '>=',
 '>>',
 '>>=',
 '@',
 '@=',
 'False',
 'True',
 '[',
 '\\',
 ']',
 '^',
 '^=',
 '_',
 '__future__',
 '_compound_statement',
 '_simple_statement',
 'aliased_import',
 'and',
 'as',
 'assert',
 'assignment',
 'async',
 'attribute',
 'augmented_assignment',
 'await',
 'binary_operator',
 'boolean_operator',
 'break',
 'break_statement',
 'call',
 'case',
 'class',
 'class_definition',
 'comment',
 'continue',
 'continue_statement',
 'def',
 'default_parameter',
 'del',
 'elif',
 'elif_clause',
 'ellipsis',
 'else',
 'else_clause',
 'escape_interpolation',
 'escape_sequence',
 'except',
 'except*',
 'exec',
 'expression',
 'finally',
 'float',
 'for',
 'for_in_clause',
 'for_statement',
 'format_expression',
 'from',
 'function_definition',
 'future_import_statement',
 'global',
 'identifier',
 'if',
 'if_statement',
 'import',
 'import_prefix',
 'import_statement',
 'in',
 'integer',
 'interpolation',
 'is',
 'is not',
 'keyword_argument',
 'keyword_separator',
 'lambda',
 'line_continuation',
 'match',
 'match_statement',
 'named_expression',
 'none',
 'nonlocal',
 'not',
 'not in',
 'not_operator',
 'or',
 'pair',
 'parameter',
 'pass',
 'pass_statement',
 'pattern',
 'positional_separator',
 'primary_expression',
 'print',
 'raise',
 'return',
 'string_end',
 'string_start',
 'subscript',
 'try',
 'type',
 'type_conversion',
 'typed_default_parameter',
 'unary_operator',
 'while',
 'while_statement',
 'wildcard_import',
 'with',
 'with_item',
 'yield',
 '{',
 '|',
 '|=',
 '}',
 '~', 
 'other'] 



NODE_TYPES = {k : i for i, k in enumerate(NODE_TYPES)}

class ASTGuidedPooling(nn.Module):
    def __init__(self, embedding_dim: int):
        super(ASTGuidedPooling, self).__init__()
        self.parser = Parser(Language(tspython.language()))
        self.max_ast_nodes = len(NODE_TYPES) + 1
        self.embedding_dim = embedding_dim
        self.node_weights = nn.Parameter(torch.ones(len(NODE_TYPES), embedding_dim))

    
    def extract_leaf_node_types(self, node, node_types):
        if len(node.children) == 0:
            node_types[(node.start_byte, node.end_byte)] = node.type  
        else:
            for child in node.children:
               self.extract_leaf_node_types(child, node_types)

    def parse_code(self, code):
        node_types = {}
        root_node = self.parser.parse(bytes(code, 'utf8')).root_node
        self.extract_leaf_node_types(root_node, node_types)
        return node_types
    
    def align_llm_to_ast(self, ast_mapped_tokens, llm_tokenizer_token_offsets):
        aligned_tokens = []
        for llm_offset in llm_tokenizer_token_offsets:
            start, end = llm_offset
            if start == end:
                aligned_tokens.append(NODE_TYPES['other'])
                continue
            for (ast_start, ast_end), token_type in ast_mapped_tokens.items():
                if ast_start <= start < ast_end:
                    aligned_tokens.append(NODE_TYPES[token_type if token_type in NODE_TYPES else 'other'])
                    break

        return aligned_tokens

    def forward(self, encoded_sequence: torch.Tensor, attention_mask, tokenizer_offsets, code) -> torch.Tensor:
        alligned_tokens = [self.align_llm_to_ast(self.parse_code(c), tokenizer_offsets[i:i+1]) for i, c in enumerate(code)]
        node_indices = torch.tensor(alligned_tokens, dtype=torch.long, device=encoded_sequence.device)
        ast_node_weights = self.node_weights[node_indices]
        token_embeddings = encoded_sequence * ast_node_weights.unsqueeze(0)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)