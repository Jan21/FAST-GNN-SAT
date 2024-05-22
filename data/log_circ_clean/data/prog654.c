
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*39 + nondet_char()*38 != 166);
}
