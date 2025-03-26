import unittest
from .transformations import perturb

class TestPerturbFunction(unittest.TestCase):
    def test_case_1(self):
        og_code = """
def new_function(x):
    return x
def old_function(a, b):
    new_function(3)
    variable = 10
    print(variable)
    return variable

old_function()
"""
        compile(og_code, '<string>', 'exec')
        results = perturb(og_code, transformations= ['rename_function'], psi = 1, depth=1, samples=1)
        
        print(results[0]['result'])
        self.assertTrue(results[0]['changed'])