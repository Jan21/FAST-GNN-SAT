/*
 * File:   Reader.hh
 *
 * Created on January 12, 2011, 4:19 PM
 */

#ifndef READER_HH
#define READER_HH
#include <stdio.h>
#include <zlib.h>

#include <iostream>
#include <utility>

#include "parse_utils.h"
using std::istream;
class Reader {
 public:
  Reader(gzFile& zf);
  Reader(StreamBuffer& zipStream);
  Reader(istream& stream);
  Reader(const Reader& orig);
  virtual ~Reader();
  int operator*();
  void operator++();
  void skip_whitespace();
  inline size_t get_line_number();

 private:
  size_t lnn;
  StreamBuffer* zip = nullptr;
  istream* s;
  int c;
};

inline size_t Reader::get_line_number() { return lnn; }
#endif /* READER_HH */

