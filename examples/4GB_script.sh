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


srun --label python -u sbatch_ssd_gnn_train.py --path {IGB_PATH}  --dataset_size medium  --data IGB  --fan_out "5,5" --num_layers 2 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type sage --cache_backend nvshmem > COALA_SAGE_5_5.txt
srun --label python -u sbatch_ssd_gnn_train.py --path {IGB_PATH}  --dataset_size medium  --data IGB  --fan_out "10,5,5" --num_layers 2 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type sage --cache_backend nvshmem > COALA_SAGE_10_5_5.txt
srun --label python -u sbatch_ssd_gnn_train.py --path {IGB_PATH}  --dataset_size medium  --data IGB  --fan_out "5,5" --num_layers 2 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type gat --cache_backend nvshmem > COALA_GAT_5_5.txt
srun --label python -u sbatch_ssd_gnn_train.py --path {IGB_PATH}  --dataset_size medium  --data IGB  --fan_out "10,5,5" --num_layers 2 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type gat --cache_backend nvshmem > COALA_GAT_10_5_5.txt


srun --label python -u sbatch_ssd_gnn_train.py --path {OGB_PATH}   --data OGB  --fan_out "5,5" --num_layers 2 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type sage --cache_backend nvshmem > OGB_COALA_SAGE_5_5.txt
srun --label python -u sbatch_ssd_gnn_train.py --path {OGB_PATH}   --data OGB  --fan_out "10,5,5" --num_layers 2 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type sage --cache_backend nvshmem > OGB_COALA_SAGE_10_5_5.txt
srun --label python -u sbatch_ssd_gnn_train.py --path {OGB_PATH}   --data OGB  --fan_out "5,5" --num_layers 2 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type gat --cache_backend nvshmem > OGB_COALA_GAT_5_5.txt
srun --label python -u sbatch_ssd_gnn_train.py --path {OGB_PATH}   --data OGB  --fan_out "10,5,5" --num_layers 2 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type gat --cache_backend nvshmem > OGB_COALA_GAT_10_5_5.txt


python output_txt_parser.py COALA_SAGE_5_5.txt IGB_SAGE_5_5 > 4g_coala_out.txt
python output_txt_parser.py COALA_SAGE_10_5_5.txt IGB_SAGE_10_5_5 >> 4g_coala_out.txt
python output_txt_parser.py COALA_GAT_5_5.txt IGB_GAT_5_5 >> 4g_coala_out.txt
python output_txt_parser.py COALA_GAT_10_5_5.txt IGB_GAT_10_5_5 >> 4g_coala_out.txt

python output_txt_parser.py OGB_COALA_SAGE_5_5.txt OGB_SAGE_5_5 >> 4g_coala_out.txt
python output_txt_parser.py OGB_COALA_SAGE_10_5_5.txt OGB_SAGE_10_5_5 >> 4g_coala_out.txt
python output_txt_parser.py OGB_COALA_GAT_5_5.txt OGB_GAT_5_5 >> 4g_coala_out.txt
python output_txt_parser.py OGB_COALA_GAT_10_5_5.txt OGB_GAT_10_5_5 >> 4g_coala_out.txt

rm OGB_COALA_SAGE_5_5.txt OGB_COALA_SAGE_10_5_5.txt OGB_COALA_GAT_5_5.txt OGB_COALA_GAT_10_5_5.txt COALA_GAT_10_5_5.txt COALA_GAT_5_5.txt COALA_SAGE_10_5_5.txt COALA_SAGE_5_5.txt