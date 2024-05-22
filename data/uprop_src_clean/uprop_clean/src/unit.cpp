/*
 * File:  Unit.cc
 * Created on:  Thu Dec 29 22:29:15 CEST 2011
 */
#include "unit.h"

#ifdef NDEBUG
#define TRACE(verbosity, command)
#else
#define TRACE(verbosity, command)                                              \
    do {                                                                       \
        if (verbose >= verbosity) {                                            \
            command                                                            \
        }                                                                      \
    } while (false)
#endif

static int verbose = 3;

Unit::Unit(const CNF &cnf) : conflict(false), que_head(0) { add_clauses(cnf); }

Unit::~Unit() {
    for (auto w : watches)
        delete w;
}

bool Unit::add_clauses(const CNF &cnf) {
    for (const LitSet &c : cnf)
        if (!add_clause(c))
            return false;
    return true;
}

bool Unit::is_failed_lit(Lit l) {
    assert(!conflict);
    assert(que_head == trail.size());
    TRACE(1, std::cerr << l << " testing" << std::endl;);
    TRACE(2, if (value(l) != l_Undef) std::cerr << l << " already has a value "
                                                << value(l) << std::endl;);
    assert(value(l) == l_Undef);
    const auto orig = trail.size();
    const auto orig_que_head = que_head;
    schedule(l);
    propagate();
    const bool rv = conflict;
    while (trail.size() > orig) {
        TRACE(2, std::cerr << trail.back() << " popping has a value "
                           << value(trail.back()) << std::endl;);
        const size_t v = var(trail.back());
        if (v < values.size())
            values[v] = l_Undef;
        trail.pop_back();
    }
    que_head = orig_que_head;
    conflict = false;
    return rv;
}

bool Unit::assert_lit(Lit literal) {
    TRACE(1, std::cerr << literal << " asserting" << std::endl;);
    schedule(literal);
    return propagate();
}

void Unit::eval(CNF &cnf) {
    if (conflict) {
        cnf.push_back(LitSet());
        return;
    }

    LiteralVector ls;
    for (size_t i = 0; i < clauses.size(); ++i) {
        const LiteralVector &clause = clauses[i];
        ls.clear();
        bool taut = false;
        for (Lit literal : clause) {
            const lbool v = value(literal);
            if (v == l_Undef) {
                ls.push_back(literal);
            } else if (v == l_True) {
                taut = true;
                break;
            } else {
                assert(v == l_False);
                dirty_clauses[i] = true;
            }
        }

        if (!taut) {
            if (dirty_clauses[i]) {
                cnf.push_back(LitSet::mk(ls));
            } else {
                const LitSet &orig_cl = original_clauses[i];
                /* assert (LitSet(ls).equal(orig_cl)); */
                cnf.push_back(orig_cl);
            }
        }
    }
}

bool Unit::add_clause(const LitSet &clause) {
    clauses.resize(clauses.size() + 1);
    const size_t clause_index = clauses.size() - 1;
    LiteralVector &ls = clauses[clause_index];
    ls.clear();
    bool taut = false;
    bool change = false;
    for (Lit literal : clause) {
        const lbool v = value(literal);
        if (v == l_Undef)
            ls.push_back(literal);
        else {
            change = true;
            if (v == l_True) {
                taut = true;
                break;
            }
        }
    }
    if (taut) {
        clauses.pop_back(); // discard
        return true;
    }

    if (ls.size() == 1) { // unit
        schedule(ls[0]);
        clauses.pop_back(); // discard
        return true;
    }

    if (ls.size() == 0) { // confl
        clauses.pop_back();
        conflict = true;
        return false;
    }

    assert(clause.size() >= 2);
    original_clauses.push_back(clause);
    assert(dirty_clauses.size() == clause_index);
    dirty_clauses.resize(clause_index + 1, false);
    if (change)
        dirty_clauses[clause_index] = true;
    // setup watches
    watch(~ls[0], clause_index);
    watch(~ls[1], clause_index);
    return true;
}

bool Unit::propagate() {
    while (!conflict && que_head < trail.size()) {
        if (!propagate(trail[que_head++]))
            conflict = true;
    }
    if (0) std::cout << "c propagated trail size: " << trail.size() << std::endl;
    if (0) std::cout << "c propagated values: " << num_propagated << std::endl;
    return !conflict;
}

void Unit::watch(Lit literal, size_t clause_index) {
    const size_t index = literal_index(literal);
    if (index >= watches.size())
        watches.resize(index + 1, nullptr);
    if (watches[index] == nullptr)
        watches[index] = new vector<size_t>();
    watches[index]->push_back(clause_index);
}

void Unit::set_value(Var variable, lbool value) {
    const size_t index = (size_t)variable;
    if (index >= values.size())
        values.resize(index + 1, l_Undef);
    values[index] = value;
}

bool Unit::propagate(Lit literal) {
    const Var variable = var(literal);
    const lbool literal_value = sign(literal) ? l_False : l_True;
    if (value(variable) != l_Undef)
        return (literal_value == value(variable));
    set_value(variable, literal_value);
    const Lit false_literal = ~literal;
    const size_t li = literal_index(literal);
    if (li >= watches.size() || watches[li] == nullptr)
        return true; // the literal does not watch anything
    num_propagated++;

    // ws are the clauses watched by literal, since the literal may stop
    // watching some clauses, the vector may shrink, we do that via two indices
    auto &ws = *(watches[li]);
    size_t read_index = 0;  // index from which we are reading from ws
    size_t write_index = 0; // index to which we are writing in ws
    bool return_value = true;
    while (read_index < ws.size()) {
        const size_t clause_index = ws[read_index++];
        assert(clause_index < clauses.size());
        LiteralVector &clause = clauses[clause_index];
        if (clause[0] == false_literal) { // swap the  literal that just became
                                          // false into position 1
            clause[0] = clause[1];
            clause[1] = false_literal;
        }
        assert(clause[1] == false_literal);
        if (value(clause[0]) == l_True) {
            ws[write_index++] = clause_index;
            continue; // clause already true
        }
        size_t new_watch_index = 0;
        for (size_t i = 2; i < clause.size(); ++i) { // find a new watch
            if (value(clause[i]) != l_False) {
                new_watch_index = i;
                break;
            }
        }

        if (new_watch_index > 0) { // new watch found
            // swap the new watch into index 1, and the old watch literal into
            // its index
            assert(new_watch_index > 1);
            watch(~clause[new_watch_index], clause_index);
            clause[1] = clause[0];
            clause[0] = clause[new_watch_index];
            clause[new_watch_index] = false_literal;
        } else { // no new watch found
            ws[write_index++] = clause_index;
            if (value(clause[0]) == l_False) { // conflict
                TRACE(1, std::cerr << "conflict on " << clauses[clause_index]
                                   << std::endl;);
                while (read_index <
                       ws.size()) // copy the rest of the watched clauses
                    ws[write_index++] = ws[read_index++];
                return_value = false;
                break;
            }
            // unit clause
            assert(value(clause[0]) == l_Undef);
            schedule(clause[0]);
        }
    }
    if (write_index < ws.size())
        ws.resize(write_index);
    return return_value;
}
