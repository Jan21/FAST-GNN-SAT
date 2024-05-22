/*
 * File:   auxiliary.hh
 *
 * Created on October 12, 2011
 */
#ifndef AUXILIARY81089_HH
#define AUXILIARY81089_HH
#include <sys/time.h>

#include <iostream>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifndef __MINGW32__
#include <sys/resource.h>
#endif
#include <assert.h>
#include <string.h>

#include <iomanip>

#define OUT std::cout

#define LOG(lev, code) \
  do {                 \
    if (verb >= lev) { \
      code             \
    }                  \
  } while (0)

#ifdef LOGLRN
#define LOG_LRN(code) \
  do {                \
    code              \
  } while (0)
#else
#define LOG_LRN(code)
#endif

#define lout (std::cerr)

#ifdef __TIMING__
#define __TIMECODE(code)                                                \
  const double _tm1245 = read_cpu_time();                               \
  std::cerr << "Starting timer: " << __FILE__ << ":" << __LINE__ << " " \
            << std::endl;                                               \
  code std::cerr << "time: " << __FILE__ << ":" << __LINE__ << " "      \
                 << std::fixed << std::setprecision(2)                  \
                 << (read_cpu_time() - _tm1245) << std::endl;
#else
#define __TIMECODE(code) code
#endif

#define __PL (std::cerr << __FILE__ << ":" << __LINE__ << std::endl).flush();
#define DPRN(msg)                                                        \
  (std::cerr << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl) \
      .flush();
/* #define FOR_EACH(index,iterated)\ */
/*   for (auto index = (iterated).begin(); index != (iterated).end();++index) */

#define SATSPC Minisat

#ifdef __MINGW32__
inline double read_cpu_time() { return 0; }
#else
inline double read_cpu_time() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000;
}
#endif

#define VERIFY(c)        \
  do {                   \
    if (!(c)) assert(0); \
  } while (0)

typedef unsigned int uint;

namespace std {
template <>
struct hash<std::pair<size_t, size_t> > {
  inline size_t operator()(const std::pair<size_t, size_t>& p) const {
    return p.first ^ p.second;
  }
};
}  // namespace std

int strtonum(const char* s, int* const n);

#endif /* AUXILIARY_HH */

