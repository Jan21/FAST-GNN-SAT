
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*32 + nondet_char()*35 != 24);
}
