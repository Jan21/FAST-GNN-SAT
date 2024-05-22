/*
 * File:   qtypes.hh
 *
 * Created on January 12, 2011, 5:32 PM
 */

#pragma once
#include "LitSet.h"
#include "minisat/core/SolverTypes.h"
#include "minisat_auxiliary.h"
#include <iostream>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

typedef std::vector<Minisat::Var> VariableVector;

typedef std::unordered_map<int, int> Int2Int;
typedef std::unordered_map<int, Lit> Int2Lit;
typedef std::vector<LitSet> CNF;

ostream &operator<<(std::ostream &outs, const CNF &f);
ostream &operator<<(std::ostream &outs, const VariableVector &ls);
std::ostream &print_dimacs(const CNF &in, std::ostream &o);
void serialize_variables(const CNF &in, CNF &out);

