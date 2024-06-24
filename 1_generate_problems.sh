#!/bin/sh

python generate_random_sat.py \
    --n_pairs 25000 \
    --out_dir temp/cnfs/selsam_3_40/train \
    --min_n 5 \
    --max_n 40 
python generate_random_sat.py \
    --n_pairs 1000 \
    --out_dir temp/cnfs/selsam_3_40/val \
    --min_n 40 \
    --max_n 40 

python generate_random_sat.py \
    --n_pairs 1000 \
    --out_dir temp/cnfs/selsam_3_40/test \
    --min_n 40 \
    --max_n 40 


