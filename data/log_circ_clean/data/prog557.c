
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*59 + nondet_char()*4 != 207);
}
