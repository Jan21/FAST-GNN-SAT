
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*52 + nondet_char()*36 != 31);
}
