#!/bin/bash
#
# File:  run_cbmc.sh
# Created on:  Mon May 29 14:50:50 CEST 2023
#
f="${1}"
cbmc --dimacs --outfile ${f}.cnf ${f} 2>/dev/null >/dev/null
