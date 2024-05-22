
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*48 + nondet_char()*12 != 29);
}
