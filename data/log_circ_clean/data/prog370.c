
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*46 + nondet_char()*1 != 26);
}
