
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*53 + nondet_char()*51 != 28);
}
