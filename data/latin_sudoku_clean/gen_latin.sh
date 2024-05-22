#!/bin/bash
#
# File:  gen.sh
# Created on:  Thu Jun 22 06:02:19 PM UTC 2023
#
#
sed -u
DATA_DIR=data_latin
UP_DATA_DIR=up_data_latin
CORES=64
COUNT=500
SAMPLE=200
for ORDER in 8 9; do
  DD=${DATA_DIR}_${ORDER}
  UDD=${UP_DATA_DIR}_${ORDER}
  SSUDD=${UP_DATA_DIR}_SAT_${SAMPLE}_${ORDER}
  rm -rfv $DD $UDD $SSUDD
  mkdir -p $DD
  mkdir -p $UDD
  mkdir -p $SSUDD
  ./gen_latin_critical.py --count ${COUNT} --order ${ORDER} --data ${DD}
  echo "removing duplicates"
  fdupes -q $DD | python3 ./rmdupes.py
  ls ${DD}/*.cnf | parallel -j ${CORES} ./run_up.sh ${UDD}
  echo "removing duplicates"
  fdupes -q ${UDD} | python3 ./rmdupes.py
  ls ${UDD}/*_sat.cnf | shuf -n ${SAMPLE} | while read f; do echo sampling $f; cp -v $f ${SSUDD}/; done
done
