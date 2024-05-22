
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*20 + nondet_char()*55 != 178);
}
