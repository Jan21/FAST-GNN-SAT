
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*14 + nondet_char()*51 != 49);
}
