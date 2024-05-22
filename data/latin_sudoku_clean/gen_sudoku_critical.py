#!/usr/bin/env python3
""" Generation of partially filled in Sudoku squares """
# File:  gen_sudoku_critical.py
# Created on:  Thu Jun 22 17:43:37 CEST 2023
import random
import sys
import os
import argparse

from pysat.solvers import Solver
from pysat.formula import IDPool

def var2str(ids, var):
    """ Print nicely a CNF var. """
    obj = ids.obj(var)
    return obj if obj is not None else '#' + str(var)

def lit2str(ids, lit):
    """ Print nicely a CNF literal. """
    return ('-' if lit < 0 else '+') + "{" + var2str(ids, abs(lit)) + "}"

def prn_clause(ids, clause):
    """ Print nicely a CNF clause. """

    if isinstance(clause, int):
        print('ERROR:', lit2str(ids, clause))
    else:
        print('[', ' '.join([lit2str(ids, lit) for lit in clause]), ']')

def print_cnf(ids, cnf):
    """ Print nicely a CNF. """
    print('[')
    for clause in cnf:
        prn_clause(ids, clause)
    print(']')

def atmost1(lits):
    """ Return CNF that at most one literal is true. """
    cnf = []
    for i, lit1 in enumerate(lits):
        for lit2 in lits[i+1:]:
            cnf.append([-lit1, -lit2])
    return cnf

class Enc:
    """ Class calculating the encoding in CNF. """

    def __init__(self, ids, small):
        self.ids = ids
        self.small = small
        self.order = self.small * self.small
        self.val = lambda row, col, digit, copy: ids.id(f'v {row} {col} {digit} {copy}')

    def val(self, row, col, digit, copy):
        assert 0 <= row < self.order
        assert 0 <= col < self.order
        assert 0 <= digit < self.order
        return ids.id(f'v {row} {col} {digit} {copy}')

    def diff(self, row, col, digit, copy1, copy2):
        assert 0 <= row < self.order
        assert 0 <= col < self.order
        assert 0 <= digit < self.order
        return self.ids.id(f'd {row} {col} {digit} {copy1} {copy2}')

    def enc(self, copy):
        """ Encode Sudoku into CNF for one copy of the square. """
        cval = lambda row, col, digit: self.val(row, col, digit, copy)
        rng = range(self.order)
        cnf = []
        # there is at least one digit in each cell
        cnf += [[ cval(row, col, digit) for digit in rng ] for row in rng for col in rng]
        # each digit appears at most one in each row and column
        for digit in rng:
            for row in rng:
                cnf += atmost1([cval(row, col, digit) for col in rng])
            for col in rng:
                cnf += atmost1([cval(row, col, digit) for row in rng])
        # go through boxes
        for rsmall in range(self.small):
            for csmall in range(self.small):
                # each digit appears at most one in each box
                rrng = range(rsmall * self.small, (1+rsmall) * self.small)
                crng = range(csmall * self.small, (1+csmall) * self.small)
                for digit in rng:
                    cnf += atmost1([cval(row, col, digit) for row in rrng for col in crng])
        return cnf

    def enc_diff(self, copy1, copy2):
        """ Encode that 2 copies of the square differ. """
        rng = range(self.order)
        cnf = []
        cnf += [[-self.diff(row, col, digit, copy1, copy2),
                 -self.val(row, col, digit, copy1),
                 -self.val(row, col, digit, copy2)]
                 for digit in rng for row in rng for col in rng]

        cnf += [[-self.diff(row, col, digit, copy1, copy2),
                 self.val(row, col, digit, copy1),
                 self.val(row, col, digit, copy2)]
                 for digit in rng for row in rng for col in rng]

        cnf += [[self.diff(row, col, digit, copy1, copy2) for digit in rng for row in rng for col in rng]]
        return cnf

    def mk_solution(self, model_list, copy):
        """ Make solution in the form of a hash map from position to values. """
        rng = range(self.order)
        model = {lit for lit in model_list if lit > 0}
        retv = dict()
        for row in rng:
            for col in rng:
                found = 0
                for digit in rng:
                    if self.val(row, col, digit, copy) in model:
                        found += 1
                        retv[(row,col)]=digit
                assert found == 1
        return retv

    def sol2str(self, sol, pref=''):
        """ Print solution. """
        rng = range(self.order)
        retv = ''
        for row in rng:
            retv += pref
            if (row % self.small == 0):
                retv +=  f'{2*self.order * "-"}\n{pref}'
            for col in rng:
                retv += '|' if (col % self.small == 0) else (' ' if (col > 0) else '')
                pos = (row,col)
                retv += str(sol[pos] + 1) if pos in sol else '_'
            retv += '\n'
        return retv

def gen_rnd(args, enc, cnf):
    """ Try to generate a random sudoku. """
    rng = range(enc.order)
    # generate random values
    assumptions = [enc.val(row, col, random.randint(0, enc.order - 1), 0)
                   for row in rng for col in rng]
    random.shuffle(assumptions)
    with Solver(bootstrap_with=cnf) as sat:
        # remove values until it becomes SAT
        while not sat.solve(assumptions):
            assumptions.pop()
        sol = enc.mk_solution(sat.get_model(), 0)
    print(enc.sol2str(sol))
    return sol

def write_file(outf, args, enc, cnf, sol, sub_sol):
    """ Produce the dimacs. """
    ids = enc.ids
    order = args.box * args.box
    outf.write(f'c box={args.box} order={order}\n')
    assert enc.order == order
    outf.write(enc.sol2str(sol, "c "))
    outf.write("c " + (2 * order) * '-' + " \n")
    outf.write(enc.sol2str(sub_sol, "c "))
    outf.write(f'p cnf {ids.top} {len(cnf)+len(sub_sol)}\n')
    for clause in cnf:
        outf.write(' '.join(map(str, clause)) + ' 0\n')
    for (row,col) in sub_sol:
        outf.write(str(enc.val(row, col, sub_sol[(row,col)], 0)) + ' 0\n')

def gen(args, data_id):
    """ Generate problem data_id. """
    print(data_id, '===================')
    order = args.box ** 2
    rng = range(order)
    ids = IDPool()
    enc = Enc(ids, args.box)
    assert enc.order == order
    cnf = enc.enc(0)
    # print_cnf(ids, cnf)
    sol = gen_rnd(args, enc, cnf)
    cnf += enc.enc(1)
    cnf += enc.enc_diff(0, 1)
    cells = [(row, col) for row in rng for col in rng]
    random.shuffle(cells)
    with Solver(bootstrap_with=cnf) as sat:
        i = 0
        unsat_cases = 0
        while i < len(cells):
            cells1 = cells[:i] + cells[i+1:]
            assumptions = [enc.val(row, col, sol[(row, col)], copy)
                           for (row,col) in cells1 for copy in [0,1] ]
            if sat.solve(assumptions):
                i += 1
                # sol_c0 = enc.mk_solution(sat.get_model(), 0)
                # sol_c1 = enc.mk_solution(sat.get_model(), 0)
                # print(enc.sol2str(sol_c0))
                # print(enc.sol2str(sol_c1))
            else:
                unsat_cases += 1
                cells = cells1
        assert unsat_cases > 0

    sub_sol = { (row,col):sol[(row,col)] for (row,col) in cells }
    empty_cells = [(row, col) for row in rng for col in rng if (row,col) not in sub_sol]
    empty_cell = random.choice(empty_cells)

    print(enc.sol2str(sub_sol))
    # for simplicity re-create the encoding for a single copy
    enc1 = Enc(IDPool(), args.box)
    assert enc.order == order
    cnf1 = enc1.enc(0)
    name = f"sudoku_critical_{args.seed}_{args.box}_{data_id}"
    data_dir = args.data

    # write SAT instance
    sub_sol[empty_cell] = sol[empty_cell]
    with open(f"{data_dir}/{name}_sat.cnf", "w") as outf:
        write_file(outf, args, enc1, cnf1, sol, sub_sol)

    # write UNSAT instances
    unsat_vals = list(range(0, sol[empty_cell])) \
            + list(range(1 + sol[empty_cell], order))
    unsat_vals = random.sample(unsat_vals, min(len(unsat_vals), args.umul))
    unsat_clues = sub_sol.copy()
    for unsat_val in unsat_vals:
        unsat_clues[empty_cell] = unsat_val
        with open(f"{data_dir}/{name}_unsat_{unsat_val}.cnf", "w") as outf:
            write_file(outf, args, enc1, cnf1, sol, unsat_clues)

def run_main():
    """ Run the whole program."""
    parser = argparse.ArgumentParser(description='Random generator for Latin squares.')
    parser.add_argument('-s', '--seed', default="77", help="seed for the random generator")
    parser.add_argument('--count', type=int, default=2, help="number of problems to generate")
    parser.add_argument('--box', type=int, default=2, help="size of the box (order=box*box)")
    parser.add_argument('--data', type=str, default='data',
                        help="directory to store the output data")
    parser.add_argument('--umul', type=int, default=10,
                        help="generate this number of unsats (only meaningful if less than order)")
    args = parser.parse_args()

    data_dir = args.data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    random.seed(args.seed)
    for i in range(args.count):
        gen(args, i)

if __name__ == "__main__":
    run_main()
    sys.exit(0)
