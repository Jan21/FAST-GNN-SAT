/*
 * File:   Read2Q.hh
 *
 * Created on Tue Jan 11 15:08:14
 */
#ifndef READ2Q_HH
#define	READ2Q_HH
#include <zlib.h>
#include <utility>
#include <vector>
#include <stdio.h>
#include <unordered_set>
#include "Reader.h"
#include "LitSet.h"
#include "ReadException.h"
#include "minisat_auxiliary.h"
using SATSPC::Lit;
using SATSPC::Var;

class ReadCNF {
public:
    ReadCNF(Reader& r);
    ~ReadCNF();
    void                           read();
    Var                            get_max_id() const;
    const std::vector< LitSet >&        get_clauses() const;
    bool                           get_header_read() const;
private:
    Reader&                 r;
    Var                     max_id;
    bool                    _header_read;
    std::vector< LitSet >        clause_vector;
    std::unordered_set<Var> unquantified_variables;
    void                    read_header();
    void                    read_clauses();
    void                    read_cnf_clause(Reader& in, std::vector<Lit>& lits);
    Var                     parse_variable(Reader& in);
    int                     parse_lit(Reader& in);
    void                    read_word(const char* word, size_t length);
};
#endif	/* READ2Q_HH */
