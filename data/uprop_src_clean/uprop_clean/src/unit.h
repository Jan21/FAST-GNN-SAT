/*
 * File:  Unit.hh
 * Created on:  Thu Dec 29 22:29:07 CEST 2011
 */
#ifndef UNIT_HH_24023
#define UNIT_HH_24023
#include "collections.h"
#include "minisat_auxiliary.h"

using Minisat::lbool;
class Unit {
  public:
    explicit Unit(const CNF &cnf);
    ~Unit();
    bool propagate();
    inline lbool value(Var variable) const;
    inline lbool value(Lit literal) const;
    void eval(CNF &cnf);
    bool is_failed_lit(Lit l);
    bool assert_lit(Lit literal);
    bool is_conflict() const { return conflict; }
    Var values_size() const { return values.size(); }

  private:
    bool conflict;              // a conflict was reached
    CNF original_clauses;       // original_clauses[i] corresponds to clauses[i]
    vector<bool> dirty_clauses; // dirty_clauses[i] iff clause[i] modified
    vector<vector<size_t> *> watches;
    vector<LiteralVector> clauses;
    size_t que_head;
    size_t num_propagated = 0;
    vector<Lit> trail;
    vector<lbool> values;
    void mark_dirty(size_t clause_index);
    bool is_dirty(size_t clause_index) const;
    void set_value(Var variable, lbool value);
    inline void schedule(Lit literal);
    bool propagate(Lit literal);
    bool add_clause(const LitSet &clause);
    bool add_clauses(const CNF &cnf);
    void watch(Lit literal, size_t clause_index);
    inline size_t literal_index(Lit l);
};

void Unit::schedule(Lit literal) { trail.push_back(literal); }

lbool Unit::value(Var variable) const {
    const size_t index = (size_t) variable;
    return index < values.size() ? values[index] : l_Undef;
}

lbool Unit::value(Lit literal) const {
    const lbool v = value(var(literal));
    if (v == l_Undef)
        return l_Undef;
    return (v == l_False) == sign(literal) ? l_True : l_False;
}

inline size_t Unit::literal_index(Lit l) {
    assert(var(l) >= 0);
    const size_t vi = (size_t)var(l);
    return sign(l) ? 2 * vi : 2 * vi + 1;
}
#endif /* UNIT_HH_24023 */
