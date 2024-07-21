from pysat.formula import CNF
from pysat.examples.rc2 import RC2

cnf = CNF(from_file='temp/cnfs/selsam_3_40/test/sr_n=0040_pk2=0.30_pg=0.40_t=0_sat=1.dimacs')
wcnf = cnf.weighted()

with RC2(wcnf) as rc2:
    print(rc2.compute())
    print(rc2.cost)
    cnt = 0
    for m in rc2.enumerate():
        print(m)
        cnt += 1
        if cnt > 10:
            break