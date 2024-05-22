
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*30 + nondet_char()*10 != 140);
}
