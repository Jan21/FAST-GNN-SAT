
#include<assert.h>
unsigned char nondet_char();

int main() {
  assert(nondet_char()*29 + nondet_char()*42 != 12);
}
