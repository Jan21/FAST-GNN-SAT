#!/bin/bash
#
# File:  run_up.sh
# Created on:  Thu Jun 22 06:06:27 PM UTC 2023
#
D=${1}
F=${2}
N=`basename ${F}`
./uprop ${F} >${D}/up_${N}
