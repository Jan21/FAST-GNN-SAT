
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*4 + nondet_char()*52 != 243);
}
