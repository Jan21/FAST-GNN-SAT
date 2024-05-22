/*
 * File:   MiniSatExt.hh
 *
 * Created on November 29, 2010, 5:40 PM
 */
#ifndef MINISATEXT_HH
#define	MINISATEXT_HH
#include "auxiliary.h"
#include "minisat/core/Solver.h"
#define BASE_SOLVER Solver
namespace SATSPC {
  class MiniSatExt : public BASE_SOLVER {
  public:
    inline double get_activity(Var var)        { return activity[var]; }
    inline void bump(Var var)        { varBumpActivity(var); }
    inline void new_variables(Var max_id);
    inline void new_variables(const std::vector<Var>& variables);
    inline lbool swap_pol(Var v, lbool def);
  };

  inline void MiniSatExt::new_variables(Var max_id) {
    const int target_number = (int)max_id+1;
    while (nVars() < target_number) newVar();
  }

  inline void MiniSatExt::new_variables(const std::vector<Var>& variables) {
    Var max_id = 0;
    for (Var v : variables)
      if (max_id < v) max_id = v;
    new_variables(max_id);
  }

  inline lbool MiniSatExt::swap_pol(Var v, lbool def) {
      assert(v < nVars());
      const auto o = user_pol[v];
      const auto r = o == l_Undef ? def : (o == l_True ? l_False : l_True);
      setPolarity(v, r);
      return r;
  }
} /* namespace SATSPC */
#endif /* MINISATEXT_HH */

