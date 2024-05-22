
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*37 + nondet_char()*25 != 233);
}
