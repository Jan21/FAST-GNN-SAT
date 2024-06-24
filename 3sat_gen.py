import random
import numpy as np
from pysat.solvers import Glucose3
from pysat.formula import CNF
import os


def refine_clauses(clauses):
  present_vars = sorted(list(set(np.abs(np.unique(clauses)))))
  rename_dict = {var:(ix+1) for ix, var in enumerate(present_vars)}
  refined_clauses = []
  for clause in clauses:
    add_clause = []
    for lit in clause:
      add_clause.append(int(np.sign(lit) * rename_dict[np.abs(lit)]))
    refined_clauses.append(add_clause)
  return refined_clauses, len(present_vars)

def generate_3sat(num_clauses, num_variables):
    variables = list(range(1, num_variables + 1))
    clauses = []

    for _ in range(num_clauses):
        clause = []
        clause_variables = random.sample(variables, 3)
        for var in clause_variables:
            if random.random() < 0.5:
                clause.append(var)
            else:
                clause.append(-var)
        clauses.append(clause)

    return refine_clauses(clauses) 

def generate_and_save_sat_problems(path, problem_num, num_claus, num_v):
  num_clauses, num_var = num_claus, num_v
  
  
  successful = 0
  for cnt in range(problem_num):
    
    #draw_n = random.randint(20, 110)
    #num_clauses, num_var = 4*draw_n, draw_n
    
    clauses, var_count = generate_3sat(num_clauses, num_var)
    solver = Glucose3()
    for clause in clauses:
      solver.add_clause(clause)
    res = solver.solve()
    if res:
      cnf = CNF(from_clauses=clauses)
      fname = f'cnt={cnt}_cls={num_clauses}_var={var_count}_sat=1.dimacs'
      
      cnf.to_file(os.path.join(path, fname))
      successful += 1
  print(f"Successfully created: {successful} problems")



data_path = "temp/cnfs/3sat_100_400/test"
os.makedirs(data_path, exist_ok=True)
generate_and_save_sat_problems(data_path, 100, 400, 100)