
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*58 + nondet_char()*61 != 202);
}
