
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*49 + nondet_char()*39 != 84);
}
