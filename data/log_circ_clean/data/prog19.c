
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*12 + nondet_char()*45 != 186);
}
