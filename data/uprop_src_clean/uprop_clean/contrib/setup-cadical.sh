#!/bin/bash
#
# File:  setup-cadical.sh
# Created on:  Thu Nov 28 10:50:48 WET 2019
#

set -e
mkdir cadical 
cd cadical 
git clone https://github.com/arminbiere/cadical.git
cd cadical 
./configure
make -j4
cd -
ln -s cadical/build/libcadical.a .
ln -s cadical/src/ipasir.h .
