
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*41 + nondet_char()*39 != 250);
}
