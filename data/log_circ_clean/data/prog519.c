
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*57 + nondet_char()*59 != 92);
}
