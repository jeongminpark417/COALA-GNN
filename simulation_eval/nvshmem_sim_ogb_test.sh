#!/bin/bash  -l

#SBATCH -t 00:20:00
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --account=bdht-delta-gpu
#SBATCH --ntasks-per-node=4  # Make sure this matches your desired configuration
#SBATCH --exclusive --mem=0
#SBATCH -p gpuA100x4

#SBATCH --output=output_OGB_two_A100_4_SSD.log   # Change output filename (%j will be replaced by the job ID)
#SBATCH --error=error_OGB_two_A100_4_SSD.log 

scontrol show hostname $SLURM_NODELIST > ip_config.txt

export MASTER_ADDR=$(head -n 1 ip_config.txt)  # Use the first node as master
export MASTER_PORT=$(shuf -i 10000-20000 -n 1)

export WORLD_SIZE=$SLURM_NTASKS  # Total number of processes

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"

srun python3   nvshmem_homogeneous_train_sim.py  --data OGB --path /projects/bdht/jpark346/ --in_memory 1 --batch_size 256 --dataset_size medium --page_size 512 --num_ele $((1024 * 100000000))   --epochs 2 --model_type sage --fan_out '5,5' --num_layers 2 --GPU_cache_size $((1 * 1024)) --CPU_cache_size $((0 * 1024))


