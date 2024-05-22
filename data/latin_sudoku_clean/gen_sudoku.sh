#!/bin/bash
#
# File:  gen.sh
# Created on:  Thu Jun 22 06:02:19 PM UTC 2023
#
#
#
sed -u
DATA_DIR=data_sudoku
UP_DATA_DIR=up_data_sudoku
CORES=64
COUNT=500
SAMPLE=200
for BOX in 3; do
  DD=${DATA_DIR}_${BOX}
  UDD=${UP_DATA_DIR}_${BOX}
  SSUDD=${UP_DATA_DIR}_SAT_${SAMPLE}_${BOX}
  rm -rfv $DD $UDD $SSUDD
  mkdir -p $DD
  mkdir -p $UDD
  mkdir -p $SSUDD
  ./gen_sudoku_critical.py --count ${COUNT} --box ${BOX} --data ${DD}
  echo "removing duplicates"
  fdupes -q $DD | python3 ./rmdupes.py
  ls ${DD}/*.cnf | parallel -j ${CORES} ./run_up.sh ${UDD}
  echo "removing duplicates"
  fdupes -q ${UDD} | python3 ./rmdupes.py
  ls ${UDD}/*_sat.cnf | shuf -n ${SAMPLE} | while read f; do echo sampling $f; cp -v $f ${SSUDD}/; done
done
