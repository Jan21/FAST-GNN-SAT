/*
 * File:  main.cpp
 * Created on:  Sat Mar 28 13:58:57 WET 2020
 */

#include "unit.h"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include "ReadCNF.h"
#include "auxiliary.h"
#include "minisat/core/Dimacs.h"
#include "minisat/core/Solver.h"
#include "minisat/utils/Options.h"
#include "minisat/utils/ParseUtils.h"
#include "minisat/utils/System.h"
#include <collections.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
using namespace std;
int run_cnf(const string &flafile);

static int verbose = 0;
#define TRACE(verbosity, command)                                              \
    do {                                                                       \
        if (verbose >= verbosity) {                                            \
            command                                                            \
        }                                                                      \
    } while (false)

int main(int argc, char **argv) {
#ifndef NDEBUG
  cout << "c DEBUG version." << endl;
#endif
  cout << "c uprop, v00.0, " << endl;
  const string flafile(argc > 1 ? argv[1] : "-");
  if (flafile == "-")
    cout << "c reading from standard input" << endl;
  else
    cout << "c reading from " << flafile << endl;
  return run_cnf(flafile);
}

int run_cnf(const string& flafile) {
  unique_ptr<Reader> fr;
  gzFile ff = Z_NULL;
  if (flafile.size() == 1 && flafile[0] == '-') {
    fr.reset(new Reader(cin));
  } else {
    ff = gzopen(flafile.c_str(), "rb");
    if (ff == Z_NULL) {
      cerr << "ERROR: "
           << "Unable to open file: " << flafile << endl;
      cerr << "ABORTING" << endl;
      exit(EXIT_FAILURE);
    }
    fr.reset(new Reader(ff));
  }
  ReadCNF reader(*fr);
  try {
    reader.read();
  } catch (ReadException& rex) {
    cerr << "ERROR: " << rex.what() << endl;
    cerr << "ABORTING" << endl;
    exit(EXIT_FAILURE);
  }
  if (ff != Z_NULL) gzclose(ff);
  cout << "c done reading: " << read_cpu_time() << std::endl;
  if (!reader.get_header_read()) {
    cerr << "ERROR: Missing header." << endl;
    cerr << "ABORTING" << endl;
    exit(EXIT_FAILURE);
  }

    Unit up(reader.get_clauses());
    const bool original_propagation = up.propagate();
    if (!original_propagation) {
        TRACE(1, cout << "c Original propagation already failed." << endl;);
    }
    CNF propagated;
    CNF shaken;
    up.eval(propagated);
    serialize_variables(propagated, shaken);
    /* print_dimacs(propagated, std::cout); */
    print_dimacs(shaken, std::cout);

    bool failed_lits = false;//TODO option
    if (!failed_lits)
        return EXIT_SUCCESS;

    if (!original_propagation) {
        TRACE(1, cout << "c Original propagation already failed." << endl;);
        return EXIT_SUCCESS;
    }

    bool fixpoint;
    int cycle_count = 0;
    bool all_defined;
    int failed_literal_counter = 0;
    do {
        cout << "== CYCLE " << ++cycle_count << endl;
        fixpoint = true;
        all_defined = true;
        for (Var v = 1; v <= reader.get_max_id(); v++) {
            if (up.value(v) != l_Undef) {
                cout << v << " already set to " << up.value(v) << endl;
                continue;
            }
            all_defined = false;

            bool unsatisfiable = false;
            if (up.is_failed_lit(mkLit(v))) {
                cout << "FAILED: " << mkLit(v) << endl;
                unsatisfiable = !up.assert_lit(~mkLit(v));
                fixpoint = false;
                ++failed_literal_counter;
            } else if (up.is_failed_lit(~mkLit(v))) {
                cout << "FAILED: " << ~mkLit(v) << endl;
                unsatisfiable = !up.assert_lit(mkLit(v));
                fixpoint = false;
                ++failed_literal_counter;
            }

            if (unsatisfiable) {
                cout << "UNSAT by failed literals " << endl;
                fixpoint = true;
                break;
            }
        }
    } while (!fixpoint);

    if (all_defined)
        cout << "== Unique model" << endl;

    cout << "== Failed literal counter:" << failed_literal_counter << endl;
    return EXIT_SUCCESS;
}
