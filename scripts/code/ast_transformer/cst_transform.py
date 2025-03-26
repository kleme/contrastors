import libcst as cst
import ast
import random
import nltk
from nltk.corpus import wordnet
import re
import keyword
import libcst.matchers as m
from typing import List

WORDS = list(wordnet.all_synsets())




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




class CommentCollector(cst.CSTVisitor):
    def __init__(self):
        self.comments = 0

    def visit_Comment(self, node):
        self.comments += 1

class CommentRemover(cst.CSTTransformer):
    def __init__(self, choice):
        self.count = 0
        self.choice = choice
    
    def leave_Comment(self, original_node, updated_node):
        if self.count == self.choice:
            return cst.RemoveFromParent()
        self.count += 1
        return updated_node


class IterVariableVisitor(cst.CSTVisitor):
    def __init__(self):
       self.iter_vars = []
    def  visit_For(self, node):
          try:
            self.iter_vars.append(node.target.value)
          except:
             pass
    def visit_While(self, node):
       if hasattr(node, "target"):
          self.iter_vars.append(node.target.value)


class IterVariableRenamer(cst.CSTTransformer):
    def __init__(self, selection, new_var):
      self.selection = selection
      self.new_var = new_var
      
    def leave_For(self, original_node, updated_node):
      return self._rename_loop_variables(updated_node)

    def leave_While(self, original_node, updated_node):
      return self._rename_loop_variables(updated_node)
    
    def _rename_loop_variables(self, node):
      updated_node = node
      if hasattr(node, "target") and isinstance(node.target, cst.Name) and node.target.value == self.selection:
          updated_node = node.with_changes(target= cst.Name(self.new_var))

      return updated_node.visit(VariableReferenceRenamer(self.selection, self.new_var))



class VariableReferenceRenamer(cst.CSTTransformer):
    def __init__(self, selection, new_var):
        self.selection = selection
        self.new_var = new_var
        

    def leave_Name(self, original_node, updated_node):
        if updated_node.value == self.selection:
            return updated_node.with_changes(value= self.new_var)
        return updated_node



class FunctionRenameParameterCollector(cst.CSTVisitor):
    def __init__(self):
        self.function_parameters = {}
        self.idx = 0

    def visit_FunctionDef(self, node):
        function_name = node.name.value
        parameters = [param.name.value for param in node.params.params if param not in {'self', 'cls'}]
        self.function_parameters[self.idx] = (function_name, parameters)
        self.idx += 1


class ParameterNameReplacer(cst.CSTTransformer):
      def __init__(self, selection, new_var):
          self.selection = selection
          self.new_var = new_var
          super().__init__()

      
      def leave_Param(self, node: cst.Param, updated_node: cst.Name) -> cst.Param:
        if updated_node.name.value == self.selection:
            return updated_node.with_changes(value= self.new_var)
        return updated_node
      
      
      def leave_Name(self, node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:
        if updated_node.value == self.selection:
            return updated_node.with_changes(value= self.new_var)
        
        return updated_node


class FunctionLocalParameterCollector(cst.CSTVisitor):
    def __init__(self):
        self.function_parameters = set()

    def visit_FunctionDef(self, node):
        for param in node.params.params:
           self.function_parameters.add(param)

class VariableNameVisitor(cst.CSTVisitor):
     def __init__(self, function_parameters):
          self.names = set()
          self.function_parameters = function_parameters
          super().__init__()
     
     def visit_Assign(self, node: cst.Name) -> cst.CSTNode:
          for target in node.targets:
             try:
              if target.target.value not in self.function_parameters:
                self.names.add(target.target.value)
             except:
                continue


class VariableNameReplacer(cst.CSTTransformer):
      def __init__(self, selection, new_var):
          self.selection = selection
          self.new_var = new_var
          super().__init__()

      def leave_Name(self, node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:
        if updated_node.value == self.selection:
            return updated_node.with_changes(value= self.new_var)
        
        return updated_node

class UnrollWhiles(ast.NodeTransformer):
    def __init__(self, selection):
      self.selection = selection
      self.count = 0
      self.done = False
      super().__init__()

    def visit_While(self, node):
      if self.done:
        return node
      if self.count != self.selection:
        self.count += 1
        return node
      
      self.done = True
      return ast.While(
        test=node.test,
        body=node.body + [ node, ast.Break() ],
        orelse=[]
      )
        
class SimpleStatementVisitor(cst.CSTVisitor):
     def __init__(self):
          self.count = 0
          super().__init__()
     
     def visit_SimpleStatementLine(self, node: cst.Name) -> cst.CSTNode:
          self.count += 1


class TryExceptTransformer(cst.CSTTransformer):
      def __init__(self, selection, generate_identifier):
          self.selection = selection
          self.generate_identifier = generate_identifier
          self.count = 0
          super().__init__()

      def leave_SimpleStatementLine(self, node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:
          self.count += 1
          if self.count == self.selection:
            
            new_var = self.generate_identifier()
            return cst.Try(
              body= cst.IndentedBlock([updated_node]),
              handlers=[
                  cst.ExceptHandler(
                      body= cst.IndentedBlock([
                            cst.parse_module(f"{new_var} = {random.random()} \n")
                        ])
                    )
                ]

          )

          return updated_node

class AddDeadCode(cst.CSTTransformer):
      def __init__(self, selection, generate_identifier):
          self.selection = selection
          self.generate_identifier = generate_identifier
          self.count = 0
          super().__init__()

      def leave_SimpleStatementLine(self, node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:
          self.count += 1
          if self.count == self.selection:
            new_var1 = self.generate_identifier()
            try:
              new_var2 = self.generate_identifier()
              new_code = cst.parse_module(f"""{new_var1} = 1 \nif {new_var1} != {new_var1}: \n  {new_var2} = {random.random()}""").body
            except:
               import pdb;pdb.set_trace()
            return cst.FlattenSentinel([*new_code, updated_node])
          return updated_node


class AddPrintStatements(cst.CSTTransformer):
      def __init__(self, selection, string_sampler):
          self.selection = selection
          self.string_sampler = string_sampler
          self.count = 0
          super().__init__()

      def leave_SimpleStatementLine(self, node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:
          self.count += 1
          if self.count == self.selection:
            new_code = cst.parse_module(f"print('{self.string_sampler()}')\n").body
            return cst.FlattenSentinel([updated_node, *new_code])
          return updated_node 




class BoolVisitCounter(cst.CSTVisitor):
      def __init__(self):
          self.count = 0
          super().__init__()

      def visit_Name(self, node: cst.Name) -> cst.CSTNode:
           if node.value in {'True', 'False'}:
              self.count += 1
           

class ReplaceTrueFalse(cst.CSTTransformer):
      def __init__(self, selection, generate_identifier):
          self.selection = selection
          self.generate_identifier = generate_identifier
          self.count = 0
          super().__init__()

      def leave_Name(self, node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:
          if node.value in {'True', 'False'}:
            self.count += 1
          if self.count == self.selection:
            if updated_node.value == 'True':
                identifier = self.generate_identifier()
                expr = f"{identifier} == {identifier}"
                self.count += 1
                return cst.parse_expression(expr)

            elif updated_node.value == 'False':
              identifier = self.generate_identifier()
              expr = f"{identifier} != {identifier}"
              self.count += 1
              return cst.parse_expression(expr)

          return updated_node


class FunctionNameCollector(cst.CSTVisitor):
    def __init__(self):
        self.function_names = set()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.function_names.add(node.name.value)

    def visit_Call(self, node: cst.Call) -> None:
        if isinstance(node.func, cst.Name):
            self.function_names.add(node.func.value)
            

class RenameFunctionTransformer(cst.CSTTransformer):
    def __init__(self, selection, new_var):
        self.selection = selection
        self.new_var = new_var

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        for old_name, new_name in [(self.selection, self.new_var)]:
            if original_node.name.value == old_name:
                return updated_node.with_changes(name=cst.Name(value=new_name))
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.CSTNode:
        for old_name, new_name in [(self.selection, self.new_var)]:
            if m.matches(original_node.func, m.Name(value=old_name)):
                new_func = cst.Name(value=new_name)
                return updated_node.with_changes(func=new_func)
        return updated_node