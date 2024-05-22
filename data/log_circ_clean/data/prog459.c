
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*11 + nondet_char()*29 != 17);
}
