
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*47 + nondet_char()*33 != 111);
}
