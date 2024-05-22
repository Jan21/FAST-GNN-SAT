
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*50 + nondet_char()*36 != 139);
}
