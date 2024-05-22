#!/bin/bash
#
# File:  setup-minisat.sh
# Created on:  Wed Dec 21 17:13:24 CET 2022
#

set -e
mkdir minisat 
cd minisat 
MDIR=`pwd`
git clone https://github.com/agurfinkel/minisat.git
cd minisat 
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${MDIR} ..
make -j4
make install
