
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*7 + nondet_char()*53 != 7);
}
