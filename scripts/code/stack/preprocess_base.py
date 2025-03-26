from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    def process_docstring(query):
        pass
    
    def extract_inner_docstring(node):
        pass
    
    def get_functions_iterative(root_node, functions, docstrings):
        pass 
    
    @staticmethod
    def extract_functions_with_docstrings(example, parser):
        pass 
    
    
    
    
        
    