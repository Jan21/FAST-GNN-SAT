
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*17 + nondet_char()*63 != 42);
}
