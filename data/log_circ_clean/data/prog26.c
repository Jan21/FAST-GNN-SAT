
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*55 + nondet_char()*32 != 213);
}
