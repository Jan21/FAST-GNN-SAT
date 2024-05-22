
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*15 + nondet_char()*44 != 121);
}
