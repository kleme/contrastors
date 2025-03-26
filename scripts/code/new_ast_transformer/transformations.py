import tree_sitter_python as tspython
import tree_sitter_javascript as tsjs
import tree_sitter_php as tsphp
import tree_sitter_go as tsgo
import tree_sitter_java as tsj
import tree_sitter_ruby as tsruby
from tree_sitter import Language, Parser
from collections import defaultdict

import random
from nltk.corpus import wordnet
import re
import keyword

WORDS = list(wordnet.all_synsets())

LANG2LIB = {'go': tsgo, 'python': tspython, 'php': tsphp, 'javascript': tsjs, 'java': tsj, 'ruby': tsruby}


class IdentifierGenerator:
    def __init__(self, psi, used_vars):
        self.psi = psi 
        self.used_vars = set(used_vars) 
    
    def __call__(self):
        i = 0
        identifier = []
        while i < self.psi:
            word = random.choice(WORDS).lemmas()[0].name()
            if (i == 0 and (not word[0].islower() or word in keyword.kwlist)) or re.search(r'[^a-zA-Z0-9]', word) or (word in self.used_vars):
                continue 
            identifier.append(word)
            self.used_vars.add(word)
            i += 1
        return '_'.join(identifier)


def rename(source_code, occurances, old_name, new_name):
    offset = 0
    for node in occurances:
        start_byte = node.start_byte + offset
        end_byte = node.end_byte + offset
        source_code = source_code[:start_byte] + new_name + source_code[end_byte:]
        offset += len(new_name) - len(old_name)
    
    return source_code


def traverse_tree(node, collected_items, language):
    if node.type in {'identifier', 'variable_name'}:
        if language == 'go':
            if node.type == 'short_var_declaration' or node.type == 'assignment_statement':
                for child in node.children:
                    if child.type == 'expression_list':
                        for expr in child.children:
                            if expr.type == 'identifier':
                              collected_items[expr.text.decode('utf8')].append(expr)
    
        else:
            if 'rename_variable' in collected_items and node.type in {'identifier', 'variable_name'}:
                if node.parent.type not in ('function_definition', 'call'):
                    collected_items[node.text.decode('utf8')].append(node)
        
    else:
        if language == 'javascript':
            if node.type in ('function_declaration', 'function') and node.child_by_field_name('name'):
                name_node = node.child_by_field_name('name')
                collected_items[name_node.text.decode('utf8')].append(name_node)
        
        elif language == 'java':
            if node.type in 'method_declaration' and node.child_by_field_name('name'):
                name_node = node.child_by_field_name('name')
                collected_items[name_node.text.decode('utf8')].append(name_node)
        
        elif language == 'ruby':
            if node.type in 'method_declaration' and node.child_by_field_name('name'):
                name_node = node.child_by_field_name('name')
                collected_items[name_node.text.decode('utf8')].append(name_node)
        
        elif language == 'php':
            if node.type in 'function_definition' and node.child_by_field_name('name'):
                name_node = node.child_by_field_name('name')
                collected_items[name_node.text.decode('utf8')].append(name_node)
             
        else:
            if node.type == 'function_declaration' and node.child_by_field_name('name'):
                name_node = node.child_by_field_name('name')
                collected_items[name_node.text.decode('utf8')].append(name_node)

    for child in node.children:
        traverse_tree(child, collected_items, language)

def has_error(node):
    if node.is_error:
        return True
    
    for child in node.children:
        if has_error(child):
            return True
    
    return False


def perturb(og_code, transformations = None, language = None, depth = 1, samples = 1, psi = 5):
    if language is None:
        no_error = False 
        
        try:
            compile(og_code, '<string>', 'exec')
            import pdb;pdb.set_trace()
            language = 'python'
            
            parser = Parser(Language(tspython.language()))
            tree = parser.parse(bytes(og_code, "utf8"))
            root_node = tree.root_node
        
        except:
            for lang in ['go', 'java', 'javascript', 'ruby', 'php']:
                parser = Parser(Language(LANG2LIB[lang].language() if lang != 'php' else tsphp.language_php()))
                tree = parser.parse(bytes(og_code, "utf8"))
                root_node = tree.root_node
                no_error = not has_error(root_node)
                
                if no_error:
                    language = lang
                    break 
            
            if not no_error:
                return [{'changed': False, 'result': og_code}] * samples 
    
    else:
        parser = Parser(Language(LANG2LIB[lang].language()))
        tree = parser.parse(bytes(og_code, "utf8"))
        root_node = tree.root_node
    
    
    
    collected_items = {t : defaultdict(list) for t in transformations}
    
    traverse_tree(root_node, collected_items, language)
    
    print(collected_items)
    
    consolidated = {}
    for v in collected_items.values():
        consolidated.update(v)
    
    if not consolidated:
        return [{'changed': False, 'result': og_code}] * samples
    

    results = []
    for _ in range(samples):
        orig_vars = list(consolidated.keys())
        generator = IdentifierGenerator(psi, orig_vars)
        
        selections = random.sample(orig_vars, depth)
        
        result = og_code
        for selection in selections:
            result = rename(result, consolidated[selection], selection, generator())
        
        results.append({'changed': True, 'result': result})
    
    return results
        
    
    
 
    
    
            
        
        
    