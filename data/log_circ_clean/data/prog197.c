
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*54 + nondet_char()*29 != 88);
}
