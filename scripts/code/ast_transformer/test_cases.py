import unittest
from .cst_perturb import perturb
from ..augment_data import augment_code
from contrastors.eval.codesage.nl2code_search import get_model
import torch
import copy

class TestPerturbFunction(unittest.TestCase):
    def test_case_1(self):
        og_code = """
x = True
y = False
if x:
    y = not y
        """
        results = perturb(og_code, psi = 1, int_id=2, depth=2, samples=1)
        self.assertTrue(results[0]['changed'])

    def test_case_2(self):
        og_code = """
def add(a, b):
    return a + b

result = add(10, 20)
        """
        results = perturb(og_code, psi = 1, int_id=6, depth=3, samples=1)
        self.assertTrue(results[0]['changed'])

    def test_case_3(self):
        og_code = """
class TestClass:
    def method(self, x):
        if x > 10:
            return True
        return False

obj = TestClass()
print(obj.method(5))
        """
        results = perturb(og_code, psi = 1, int_id=2, depth=3, samples=1)
        self.assertTrue(results[0]['changed'])

    def test_case_4(self):
        og_code = """
for i in range(10):
    x = 3
    print(i)
        """
        results = perturb(og_code, psi = 1, int_id=2, depth=3, samples=1)
        self.assertTrue(results[0]['changed'])

    def test_case_5(self):
        og_code = """
try:
    x = 1 / 0
except ZeroDivisionError:
    x = None
        """
        results = perturb(og_code, psi = 1, int_id=2, depth=3, samples=1)
        self.assertTrue(results[0]['changed'])

    def test_case_6(self):
        og_code = """
x = [1, 2, 3]
y = [a * 2 for a in x]
print(y)
        """
        results = perturb(og_code, psi = 1, int_id=2, depth=3, samples=1)
        self.assertTrue(results[0]['changed'])

    def test_case_7(self):
        og_code = """
def add(x, y, b):
    return x - b + y

result = add(10, 20, 3)
        """
        results = perturb(og_code, psi = 1, int_id=2, depth=3, samples=1)
        
        self.assertTrue(results[0]['changed'])

    def test_case_8(self):
        og_code = """
x = True
while x:
    x = False
        """
        results = perturb(og_code, psi = 1, int_id=3, depth=3, samples=1)
        self.assertTrue(results[0]['changed'])

    def test_case_9(self):
        og_code = """
def fib(n):
    a, b = 0, 1
    while a < n:
        print(a)
        a, b = b, a + b

fib(10)
        """
        results = perturb(og_code, psi = 1, int_id=3, depth=3, samples=1)
        self.assertTrue(results[0]['changed'])

    def test_case_10(self):
        og_code = """
def circle_area(radius):
    return 3.14 * radius ** 2

print(circle_area(5))
        """
        results = perturb(og_code, psi = 1, int_id=3, depth=1, samples=1)
        self.assertTrue(results[0]['changed'])

class Args:
    def __init__(self, model_name_or_path, tokenizer_name_or_path, model_type):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path

class TransformationConfig:
    def __init__(self):
        self.mode = "replace"
        self.transformation = "insert_print_statements"
        self.cols_to_transform = "document negatives"
        self.depth = 1
        self.samples = 1
        self.psi = 1


# class TestSpeed(unittest.TestCase):
#     def test_case_speed(self):
#         code = """
# def fib(n):
#     a, b = 0, 1
#     while a < n:
#         a, b = b, a + b

# fib(10)
#         """
#         codes = [code.replace("10", str(i)) for i in range(800000)]
#         dataset = [{'document': c, 'negatives': [c for _ in range(5)]} for c in codes]
#         args = TransformationConfig()
        
#         new_dataset = augment_code(dataset, args)
        
        
        
        #assert [d1['document'] == d2['document'] for d1, d2 in zip(dataset, new_dataset)]
        
        

        
        
    


# class TestEmbedPerturbed(unittest.TestCase):

            
#     def test_case_plot(self):
#         codes = ["""
# def fib(n):
#     a, b = 0, 1
#     while a < n:
#         print(a)
#         a, b = b, a + b

# fib(10)
#         """
#         ]
#         for m, t, p in [("/workspace/contrastors-dev/src/contrastors/ckpts/nomic-embed-text-v64-14/final_model", "nomic-ai/nomic-embed-text-v1-unsupervised", 'nomic'), 
#                         ("jinaai/jina-embeddings-v2-base-code", "jinaai/jina-embeddings-v2-base-code", 'transformers'), 
#                         ("Salesforce/codet5p-110m-embedding", "Salesforce/codet5p-110m-embedding", 'transformers')]:
#             model = get_model(Args(m, t, p), 1024)
#             cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#             for og_code in codes:
#                 for d in range(1, 2):
#                     perturbed_code = perturb(og_code, psi = 1, int_id=5, depth=3, samples=1)[0]['result']
#                     import pdb;pdb.set_trace()
#                     nl = torch.tensor(model.encode(['fib function'])[0].reshape(1, -1))
#                     original_code = torch.tensor(model.encode([og_code])[0].reshape(1, -1))
#                     perturbed_code= torch.tensor(model.encode([perturbed_code])[0].reshape(1, -1))
#                     print(cos(nl, original_code), cos(nl, perturbed_code), cos(original_code, perturbed_code))
                    
            
            
            
    

if __name__ == '__main__':
    unittest.main()