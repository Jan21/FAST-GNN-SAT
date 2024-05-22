
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*45 + nondet_char()*31 != 167);
}
