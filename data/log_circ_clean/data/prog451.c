
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*19 + nondet_char()*43 != 55);
}
