
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*10 + nondet_char()*16 != 76);
}