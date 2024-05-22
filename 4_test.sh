#!/bin/bash

#SR(40)
python test.py --datapath "temp/cnfs/selsam_3_40/test/" --checkpoint "final_model_checkpoint.ckpt" --ccpath "cluster_centers.pkl" --k 16 --num_iters 100 --dec 1

# SR(40) k=1 no dec
#python test.py --datapath "temp/cnfs/selsam_3_40/test/" --checkpoint "final_model_checkpoint.ckpt" --ccpath "cluster_centers.pkl" --k 1 --num_iters 100 --dec 0

# latin squares 8x8
#python test.py --datapath "data/latin_sudoku_clean/up_data_latin_SAT_200_8/" --checkpoint "final_model_checkpoint.ckpt" --ccpath "cluster_centers.pkl" --k 16 --num_iters 1000 --dec 1
#python test.py --datapath "data/latin_sudoku_clean/up_data_latin_SAT_200_8/" --checkpoint "final_model_checkpoint.ckpt" --ccpath "cluster_centers.pkl" --k 1 --num_iters 1000 --dec 0

# latin squares 9x9
#python test.py --datapath "data/latin_sudoku_clean/up_data_latin_SAT_200_9/" --checkpoint "final_model_checkpoint.ckpt" --ccpath "cluster_centers.pkl" --k 16 --num_iters 1000 --dec 1
#python test.py --datapath "data/latin_sudoku_clean/up_data_latin_SAT_200_9/" --checkpoint "final_model_checkpoint.ckpt" --ccpath "cluster_centers.pkl" --k 1 --num_iters 1000 --dec 0

#sudoku 9x9
#python test.py --datapath "data/latin_sudoku_clean/up_data_sudoku_SAT_200_3/" --checkpoint "final_model_checkpoint.ckpt" --ccpath "cluster_centers.pkl" --k 16 --num_iters 1000 --dec 1
#python test.py --datapath "data/latin_sudoku_clean/up_data_sudoku_SAT_200_3/" --checkpoint "final_model_checkpoint.ckpt" --ccpath "cluster_centers.pkl" --k 1 --num_iters 1000 --dec 0