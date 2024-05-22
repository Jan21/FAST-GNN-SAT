
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*3 + nondet_char()*32 != 22);
}
