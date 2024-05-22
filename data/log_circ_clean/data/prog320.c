
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*22 + nondet_char()*1 != 221);
}
