/*
 * File:  VarSet.hh
 * Created on:  Sun Nov 13 10:47:38 EST 2011
 */
#ifndef VARSET_HH_12377
#define VARSET_HH_12377
#include <vector>
#include <iterator>

class VarSet;

class const_VarIterator : public iterator<std::forward_iterator_tag, Var>
{
public:
  const_VarIterator(const VarSet& ls, size_t x);
  const_VarIterator(const const_VarIterator& mit) : ls(mit.ls), i(mit.i) {}
  inline const_VarIterator& operator++();
  inline Var operator*() const;
  bool operator==(const const_VarIterator& rhs) { assert(&ls==&(rhs.ls)); return i==rhs.i;}
  bool operator!=(const const_VarIterator& rhs) { assert(&ls==&(rhs.ls)); return i!=rhs.i;}
private:
  const VarSet&               ls;
  size_t                      i;
};


class VarSet {
public:
  typedef const_VarIterator const_iterator;
  /* VarSet(const VariableVector& variables); */
  /* VarSet(const VarVector& variables); */
  VarSet();
  /* void   add_all(const VarVector& variables); */
  inline bool add(Var v);
  inline bool get(Var v) const;
  inline void clear();
  inline size_t physical_size() const { return s.size(); }
  inline const_iterator begin() const;
  inline const_iterator end() const;
  inline size_t size() const;
  inline bool   empty() const;
  inline const std::vector<bool>& bs() const { return s; }
private:
  std::vector<bool> s;
  size_t           _size;
};

inline bool VarSet::add(Var _v) {
  assert(_v>0);
  size_t  v = (size_t)_v;
  if (s.size() <= v) s.resize(v+1,false);
  else if (s[v]) return false;
  s[v]=true;
  ++_size;
  return true;
}

inline bool   VarSet::get(Var v) const { return ((size_t)v<physical_size()) && s[v]; }
inline size_t VarSet::size() const   { return _size; }
inline bool   VarSet::empty() const  { return _size==0; }
inline void   VarSet::clear() {
   _size=0;
   s.clear();
}


inline const_VarIterator& const_VarIterator::operator++() {
  assert(i < ls.physical_size());
  ++i;
  while ((i<ls.physical_size()) && !ls.get(i)) ++i;
  return *this;
}

inline Var const_VarIterator::operator*() const {
  assert(ls.get(i));
  return (Var)i;
}

inline VarSet::const_iterator VarSet::end()   const { return const_VarIterator(*this, physical_size()); }
inline VarSet::const_iterator VarSet::begin() const { return const_VarIterator(*this, 0); }
#endif /* VARSET_HH_12377 */
