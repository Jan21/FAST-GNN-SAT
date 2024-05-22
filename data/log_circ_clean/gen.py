#!/usr/bin/env python3
# File:  gen.py
# Created on:  Tue May 23 10:33:31 CEST 2023
import random
import sys
import os
import argparse

def run_main():
    """run the whole program."""
    parser = argparse.ArgumentParser(description='Trivial random generator.')
    parser.add_argument('-s', '--seed', default="77")
    parser.add_argument('count', nargs='?', type=int, default=10)
    args = parser.parse_args()

    data_dir = r'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    random.seed(args.seed)
    for i in range(args.count):
        with open(f"{data_dir}/prog{i}.c", "w") as outf:
            outf.write(r"""
#include<assert.h>
unsigned char nondet_char();

int main() {
  """)
            r1 = random.randint(1, 63)
            r2 = random.randint(1, 63)
            r3 = random.randint(0, 255)
            outf.write(f"assert(nondet_char()*{r1} + nondet_char()*{r2} != {r3});\n")
            # outf.write(f"assert(nondet_char()*{r1} != {r3});\n")
            outf.write("}\n")

if __name__ == "__main__":
    run_main()
    sys.exit(0)
