
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*56 + nondet_char()*9 != 175);
}
