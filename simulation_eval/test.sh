#!/bin/bash  -l

#SBATCH -t 00:05:00
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH -G 1
#SBATCH -A m4301
#SBATCH --exclusive

ulimit -c unlimited
srun -n 1 -c 1 python3 homogeneous_train_test.py --dataset_size tiny --path /global/cfs/cdirs/m4301/jpark346/graph_data --data IGB --epochs 1   --model_type sage --num_layers 3 --fan_out '10,5,5' --emb_size 1024
