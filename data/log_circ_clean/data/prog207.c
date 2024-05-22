
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*60 + nondet_char()*43 != 13);
}
