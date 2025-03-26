"""
Code for perturbing Python programs. Taken and modified from https://github.com/uiuc-arc/llm-code-watermark/blob/main/lpw/program_perturb_cst.py.
"""
import ast
import libcst as cst
import astor
import random
from .cst_transform import *


    
  
  
  
  
  

def get_all_vars(module, ret_dct = False):
  function_name_visitor = FunctionNameCollector()
  module.visit(function_name_visitor)
  funciton_names = list(function_name_visitor.function_names)
  
  
  
  iter_visitor = IterVariableVisitor()
  module.visit(iter_visitor)
  iter_vars = list(iter_visitor.iter_vars)
  
  function_param_visitor = FunctionRenameParameterCollector()
  module.visit(function_param_visitor)
  function_vars = []
  for v in function_param_visitor.function_parameters.values():
    for vv in v[1]:
      function_vars.append(vv)
  
  
  param_visitor = FunctionLocalParameterCollector()
  module.visit(param_visitor)
  local_visitor = VariableNameVisitor(param_visitor.function_parameters)
  module.visit(local_visitor)
  local_vars = list(local_visitor.names)
  
  if ret_dct:
    return {**{v: RenameFunctionTransformer for v in funciton_names}, **{v: IterVariableRenamer for v in iter_vars}, **{v: ParameterNameReplacer for v in function_vars}, **{v : VariableNameReplacer for v in local_vars}}
  
  return iter_vars + function_vars + local_vars + funciton_names





def t_remove_comments(module, uid = 1, psi = 5):
   collector = CommentCollector()  
   module.visit(collector)
   
   if not collector.comments:
      return False, module
   
   choice = random.randrange(0, collector.comments)
   
   module.visit(CommentRemover(choice))
   
   return True, module
          

def t_rename(module, uid=1, psi = 5):
  vars = get_all_vars(module, True)
  
  if len(vars) == 0:
    return False, module
  
  selection = random.choice(list(vars.keys()))
  
  generate_random_identifier = IdentifierGenerator(psi, list(vars.keys()))

  transformer = vars[selection](selection, generate_random_identifier())
  module = module.visit(transformer)
  return True, module










def t_wrap_try_catch(module, uid=1, psi = 5):
  if len(module.body) == 0:
    return False, module
  

  visitor = SimpleStatementVisitor()

  module.visit(visitor)

  if visitor.count == 0:
     return False, module
  
  used_vars = get_all_vars(module)
  generate_random_identifier = IdentifierGenerator(psi, used_vars)

  transformer = TryExceptTransformer(visitor.count, generate_random_identifier)

  module = module.visit(transformer)

  return True, module

def t_add_dead_code(module, uid=1, psi = 5):
  if len(module.body) == 0:
    return False, module
      

  visitor = SimpleStatementVisitor()

  module.visit(visitor)

  if visitor.count == 0:
     return False, module
  
  used_vars = get_all_vars(module)
  generate_random_identifier = IdentifierGenerator(psi, used_vars)
  
  transformer = AddDeadCode(visitor.count, generate_random_identifier)

  module = module.visit(transformer)

  return True, module

  


def t_insert_print_statements(module, uid=1, psi = 5):
  if len(module.body) == 0:
    return False, module

  visitor = SimpleStatementVisitor()
  module.visit(visitor)

  if visitor.count == 0:
     return False, module
  
  used_vars = get_all_vars(module)
  generate_random_identifier = IdentifierGenerator(psi, used_vars)

  transformer = AddPrintStatements(random.randint(1, visitor.count), generate_random_identifier)
  module = module.visit(transformer)
  return True, module

def t_random_transform(module, uid = 1, psi = 5):
   i = random.randrange(0, 4)
   if i == 0:
      return t_insert_print_statements(module, uid, psi)
   elif i == 1:
      return t_add_dead_code(module, uid, psi)
   elif i == 2:
      return t_rename(module, uid, psi)

   return t_wrap_try_catch(module, uid, psi)

def t_replace_true_false(module, uid=1, psi = 5):
  visitor = BoolVisitCounter()

  module.visit(visitor)

  if visitor.count == 0:
     return False, module

  selection = random.randint(1, visitor.count)
  
  used_vars = get_all_vars(module)
  generate_random_identifier = IdentifierGenerator(psi, used_vars)
  
  transformer = ReplaceTrueFalse(selection, generate_random_identifier)

  module = module.visit(transformer)

  return True, module


class t_seq(object):
  def __init__(self, transforms, psi):
    self.transforms = transforms
    self.psi = psi
  def __call__(self, cur_ast):
    did_change = False
    for i,t in enumerate(self.transforms):
      changed, cur_ast = t(cur_ast, i+1, self.psi) 
      
      if changed:
        did_change = True
    return did_change, cur_ast.code




def t_identity(module, uid = 1, psi = 5):
  return True, module


def perturb(og_code, int_id = None, depth = 1, samples = 1, psi = 5):
    transforms = []
    DEPTH = depth
    NUM_SAMPLES = samples

    for s in range(NUM_SAMPLES):
      the_seq = []
      for i in range(DEPTH):
        if int_id == None:
          int_id = random.randint(1, 7)
        if int_id == 0:
          the_seq.append(t_identity)
        if int_id == 1:
          the_seq.append(t_replace_true_false)
        elif int_id == 2:
          the_seq.append(t_rename)
        elif int_id == 3:
          the_seq.append(t_add_dead_code)
        elif int_id == 4:
          the_seq.append(t_identity)
        elif int_id == 5:
          the_seq.append(t_insert_print_statements)
        elif int_id == 6:
          the_seq.append(t_wrap_try_catch)
        elif int_id == 7:
          the_seq.append(t_random_transform)

      transforms.append(('depth-{}-sample-{}'.format(DEPTH, s+1), t_seq(the_seq, psi), the_seq))
      
    results = []
    for t_name, t_func, the_seq in transforms:
        try:
            compile(og_code, '<string>', 'exec')
            #exec(og_code)
        except Exception as ex:
            results.append({'changed': False, 't_name': t_name, 'the_seq': the_seq, 'result': og_code})
            continue
        try:
           cur_ast = cst.parse_module(og_code)
        except:
          results.append({'changed': False, 't_name': t_name, 'the_seq': the_seq, 'result': og_code})
          continue
          #  og_code = astor.to_source(ast.parse(og_code))
          #  try: 
          #    cst.parse_module(og_code)
          #  except:
          #    results.append({'changed': False, 't_name': t_name, 'the_seq': the_seq, 'result': og_code})
          #    continue

       
        changed, result = t_func(cur_ast)
        results.append({'changed': changed, 't_name': t_name, 'the_seq': the_seq, 'result': result})

    return results

