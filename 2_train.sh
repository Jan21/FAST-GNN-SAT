#!/bin/bash

# incremental training
python train.py --datapath "temp/cnfs/selsam_3_40" --checkpoint "final_model_checkpoint.ckpt" --incremental 0

# non-incremental training
#python train.py --datapath "temp/cnfs/selsam_3_40" --checkpoint "final_model_checkpoint.ckpt" --incremental 1