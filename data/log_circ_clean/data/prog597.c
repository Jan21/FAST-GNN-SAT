
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*40 + nondet_char()*28 != 10);
}
