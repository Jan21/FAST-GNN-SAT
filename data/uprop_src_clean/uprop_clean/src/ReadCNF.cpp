#include "ReadCNF.h"
#include "ReadException.h"
#include <stdio.h>
#include <algorithm>
#include "parse_utils.h"
using std::max;
using SATSPC::mkLit;

ReadCNF::ReadCNF(Reader& r)
: r(r)
, max_id(0)
, _header_read(false)
{}

ReadCNF::~ReadCNF() {
}

bool ReadCNF::get_header_read() const  {return _header_read;}
Var ReadCNF::get_max_id() const {return max_id;}
const vector<LitSet>& ReadCNF::get_clauses() const          {return clause_vector;}

void ReadCNF::read_cnf_clause(Reader& in, vector<Lit>& lits) {
  int parsed_lit;
  lits.clear();
  for (;;){
      skipWhitespace(in);
      parsed_lit = parse_lit(in);
      if (parsed_lit == 0) break;
      const Var v = abs(parsed_lit);
      max_id = max(max_id, v);
      unquantified_variables.insert(v);
      lits.push_back(parsed_lit>0 ? mkLit(v): ~mkLit(v));
  }
}

Var ReadCNF::parse_variable(Reader& in) {
    if (*in < '0' || *in > '9') {
       string s("unexpected char in place of a variable: ");
       s+=*in;
       throw ReadException(s);
    }
    Var return_value = 0;
    while (*in >= '0' && *in <= '9') {
        return_value = return_value*10 + (*in - '0');
        ++in;
    }
    return return_value;
}

int ReadCNF::parse_lit(Reader& in) {
    Var  return_value = 0;
    bool neg = false;
    if (*in=='-') { neg=true; ++in; }
    else if (*in == '+') ++in;
    if ((*in < '0') || (*in > '9')) {
        string s("unexpected char in place of a literal: ");  s+=*in;
        throw ReadException(s);
    }
    while (*in >= '0' && *in <= '9') {
        return_value = return_value*10 + (*in - '0');
        ++in;
    }
    if (neg) return_value=-return_value;
    return return_value;
}


void  ReadCNF::read_header() {
    while (*r == 'c') skipLine(r);
    if (*r == 'p') {
      _header_read = true;
      skipLine(r);
    }
}

void ReadCNF::read_clauses() {
  Reader& in=r;
  vector<Lit> ls;
  for (;;){
    skipWhitespace(in);
    if (*in == EOF) break;
    if (*r == 'c') {
      skipLine(in);
      continue;
    }
    ls.clear();
    read_cnf_clause(in, ls);
    clause_vector.push_back(LitSet::mk(ls));
  }
}


void ReadCNF::read_word(const char* word, size_t length) {
  while (length) {
    if (word[0] != *r) {
        string s("unexpected char in place of: ");  s+=word[0];
        throw ReadException(s);
    }
    ++r;
    --length;
    ++word;
  }
}

void ReadCNF::read() {
  read_header();
  read_clauses();
}
