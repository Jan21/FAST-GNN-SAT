
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*21 + nondet_char()*30 != 219);
}
