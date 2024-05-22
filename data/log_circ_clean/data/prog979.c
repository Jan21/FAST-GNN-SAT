
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*24 + nondet_char()*58 != 112);
}
