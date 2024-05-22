
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*50 + nondet_char()*37 != 12);
}
