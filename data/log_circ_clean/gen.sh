#!/bin/bash
#
# File:  g.sh
# Created on:  Wed May 24 09:49:41 CEST 2023
#
#

DATA_DIR=data
CORES=4
COUNT=1000
rm -rfv $DATA_DIR
./gen.py ${COUNT}
echo "removing duplicates"
fdupes -q $DATA_DIR | python3 ./rmdupes.py
ls $DATA_DIR/*.c | parallel -j ${CORES} ./run_cbmc.sh
echo "removing duplicates"
fdupes -q $DATA_DIR | python3 ./rmdupes.py
OUT_FILE=o${RANDOM}${RANDOM}
(ls $DATA_DIR/*.c.cnf | parallel -j ${CORES} ./run_cadical.sh) | tee ${OUT_FILE}
echo -n "U:"; grep -nHc -e ' s UNSATISFIABLE$' ${OUT_FILE}
echo -n "S:"; grep -nHc -e ' s SATISFIABLE$' ${OUT_FILE}
ls ${DATA_DIR}/*.c.cnf\
  | while read f; do echo $f ===; ./uprop ${f} >${f%.cnf}.uprop.cnf; done
echo "removing duplicates"
fdupes -q $DATA_DIR | python3 ./rmdupes.py
(ls $DATA_DIR/*.uprop.cnf | parallel -j ${CORES} ./run_cadical.sh) | tee ${OUT_FILE}
echo -n "U:"; grep -nHc -e ' s UNSATISFIABLE$' ${OUT_FILE}
echo -n "S:"; grep -nHc -e ' s SATISFIABLE$' ${OUT_FILE}
grep 'p cnf 0 0' ${DATA_DIR}/*.cnf | cut -f1 -d: | xargs rm -fv
grep 'p cnf 0 1' ${DATA_DIR}/*.cnf | cut -f1 -d: | xargs rm -fv
(ls $DATA_DIR/*.uprop.cnf | parallel -j ${CORES} ./run_cadical.sh) | tee ${OUT_FILE}
echo -n "U:"; grep -nHc -e ' s UNSATISFIABLE$' ${OUT_FILE}
echo -n "S:"; grep -nHc -e ' s SATISFIABLE$' ${OUT_FILE}
mkdir sample
grep -e ' s SATISFIABLE$' ${OUT_FILE} | cut -d ':' -f1 | shuf -n 200 | while read f; do echo sampling $f; cp -v $f sample/; done
# rm -f ${OUT_FILE}
