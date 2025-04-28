#!/bin/bash
#SBATCH --output=coala-gnn-4x4-igb-16G-%j.out
#SBATCH --error=coala-gnn-4x4-igb-16G-%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00         # max runtime
#SBATCH --partition=gpuA100x4         # adjust to your cluster
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --job-name=COALA_GNN_4x4_cache_bench


export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
# Run your script using mpirun or srun

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=12345

# Optional: log to debug
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_PROCID=$SLURM_PROCID"


srun --label python -u sbatch_ssd_gnn_train.py --path {IGB_PATH}  --dataset_size medium  --data IGB  --fan_out "5,5" --num_layers 2 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type sage --cache_backend nvshmem > NVSHMEM_CACHE_out_5_5.txt
srun --label python -u sbatch_ssd_gnn_train.py --path {IGB_PATH}  --dataset_size medium  --data IGB  --fan_out "5,5" --num_layers 2 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type sage --cache_backend nccl > NCCL_CACHE_out_5_5.txt
srun --label python -u sbatch_ssd_gnn_train.py --path {IGB_PATH}  --dataset_size medium  --data IGB  --fan_out "5,5" --num_layers 2 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type sage --cache_backend isolated > ISOLATED_CACHE_out_5_5.txt

srun --label python -u sbatch_ssd_gnn_train.py --path {IGB_PATH}  --dataset_size medium  --data IGB  --fan_out "10,5,5" --num_layers 3 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type sage --cache_backend nvshmem > NVSHMEM_CACHE_out_10_5_5.txt
srun --label python -u sbatch_ssd_gnn_train.py --path {IGB_PATH}  --dataset_size medium  --data IGB  --fan_out "10,5,5" --num_layers 3 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type sage --cache_backend nccl > NCCL_CACHE_out_10_5_5.txt
srun --label python -u sbatch_ssd_gnn_train.py --path {IGB_PATH}  --dataset_size medium  --data IGB  --fan_out "10,5,5" --num_layers 3 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type sage --cache_backend isolated > ISOLATED_CACHE_out_10_5_5.txt

python output_txt_parser.py NVSHMEM_CACHE_out_5_5.txt COALA_5_5 > cache_bench_out.txt
python output_txt_parser.py NCCL_CACHE_out_5_5.txt NCCL_5_5 >> cache_bench_out.txt
python output_txt_parser.py ISOLATED_CACHE_out_5_5.txt ISOLATED_5_5 >> cache_bench_out.txt

python output_txt_parser.py  NVSHMEM_CACHE_out_10_5_5.txt COALA_10_5_5 >> cache_bench_out.txt
python output_txt_parser.py  NCCL_CACHE_out_10_5_5.txt NCCL_10_5_5 >> cache_bench_out.txt
python output_txt_parser.py  ISOLATED_CACHE_out_10_5_5.txt ISOLATED_10_5_5 >> cache_bench_out.txt


rm NVSHMEM_CACHE_out_5_5.txt NCCL_CACHE_out_5_5.txt ISOLATED_CACHE_out_5_5.txt NVSHMEM_CACHE_out_10_5_5.txt NCCL_CACHE_out_10_5_5.txt ISOLATED_CACHE_out_10_5_5.txt
