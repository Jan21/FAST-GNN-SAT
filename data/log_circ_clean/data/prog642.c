
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*17 + nondet_char()*16 != 236);
}
