
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*38 + nondet_char()*47 != 81);
}
