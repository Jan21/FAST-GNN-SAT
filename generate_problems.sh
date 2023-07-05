#!/bin/sh

python generate_random_sat.py \
    --n_pairs 10000 \
    --out_dir temp/cnfs/selsam_3_10/train \
    --min_n 3 \
    --max_n 10 
python generate_random_sat.py \
    --n_pairs 5000 \
    --out_dir temp/cnfs/selsam_3_10/test \
    --min_n 3 \
    --max_n 10 