#!/bin/sh

python generate_random_sat.py \
    --n_pairs 10000 \
    --out_dir temp/cnfs/selsam_40_40/train \
    --min_n 40 \
    --max_n 40 
python generate_random_sat.py \
    --n_pairs 5000 \
    --out_dir temp/cnfs/selsam_40_40/test \
    --min_n 40 \
    --max_n 40 