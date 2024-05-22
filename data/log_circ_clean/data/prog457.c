
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*35 + nondet_char()*43 != 63);
}
