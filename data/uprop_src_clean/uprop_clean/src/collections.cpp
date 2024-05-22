/*
 * File:  collections.cpp
 * Created on:  Thu May 25 17:01:28 CEST 2023
 */
#include "collections.h"

#include <LitSet.h>
#include <algorithm>
#include <ostream>
#include <unordered_map>
#include <unordered_set>

#include "minisat/core/SolverTypes.h"

std::ostream &print_dimacs(const CNF &in, std::ostream &o) {
    Var maxid = 0;
    for (const auto &cl : in)
        for (const auto &l : cl)
            maxid = std::max(maxid, var(l));

    o << "p cnf " << maxid << " " << in.size() << std::endl;
    for (const auto &cl : in) {
        for (const auto &l : cl)
            o << l << " ";
        o << "0\n";
    }
    return o;
}

void serialize_variables(const CNF &in, CNF &out) {
    std::unordered_set<Var> used;
    std::unordered_map<Var, Var> remap;
    for (const auto &cl : in)
        for (const auto &l : cl)
            used.insert(var(l));
    Var nv = 0;
    for (const auto v : used)
        remap[v] = ++nv;
    LiteralVector ls;
    for (const auto &cl : in) {
        ls.clear();
        for (const auto &l : cl) {
            const auto v = var(l);
            const auto mv = remap.at(v);
            ls.push_back(mkLit(mv, sign(l)));
        }
        out.push_back(LitSet::mk(ls));
    }
}
