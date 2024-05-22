
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*5 + nondet_char()*7 != 168);
}
