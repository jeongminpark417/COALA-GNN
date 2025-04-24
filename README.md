# COALA-GNN: Color-based Affinity-aware Cache and Distribution Strategy for Scalable Distributed Large-Scale GNN Training

This repository contains the implementation of the COALA-GNN, an open-source implementation for accelerating distributed large-scale Graph Neural Network (GNN) workloads. Contributions to the codebase are most welcome.

## Artifact Setup
### Hardware
We require NVIDIA A100 GPU or more recent GPUs (such as H100) to cover the essential functionality.

### Software
The following lists are the required software to succesfully build COALA-GNN
- CUDA Toolkit (higher than 12.2 version)
- DGL (version 2.5 is used)
- Pybind11
- Pytorch (version 2.2.0 is used)
- NVSHMEM
- MPI

  

## Get Started
### Installing COALA-GNN
To use the COALA-GNN framework, users need to set up the environment first.

```
git submodule update --init --recursive
```

To use storage-based GNN training, BaM system should be correctly set up. Run the following command to create BaM library.

```
cd bam
mkdir build
cmake .. && make -j
```

Once the BaM system is set up, users need to create Pybind modules for COALA-GNN. Run the following command to install pybind11 modules:

```
cd COALA_GNN_Modules
mkdir build
cmake .. && make -j
cd COALA_GNN_Pybind
pip install .
```

Afterwards, users can set up the Python interface for COALA-GNN Dataloader by running the following command from the root directory:

```
cd COALA-GNN-Setup
pip install .
```

### Create Metadata
COALA-GNN requires node color information, so users must run the preprocessing step to generate this information by running the following command from the root directory:
```
cd examples
cd color_info_gen
python generate_color_data.py --path  $IGB_DATASET_PATH  --dataset_size medium  --data IGB --out_path $IGB_DATASET_COLOR_PATH
python generate_color_data.py --path  $OGB_DATASET_PATH   --data OGB --out_path $OGB_DATASET_COLOR_PATH
```
The COLOR_PATH set to be the same as DATASET_PATH in the example script for the GNN training, so make sure that all `color.npy`, `topk.npy` and `score.npy` are stored in the same dataset directory.

### Create CSC graph format
In DGL framework, when the graph data strucuture is created via COO format, the CSC format is generated before running graph sampling process. However, COALA-GNN minimizes the CPU usage, the graph strucutre data is pinned in shared CPU memory and all processes in the same node shares mappes the same shared CPU memory to their memory address space to enable GPU threads to access the data via UVA (Unified Virtual memory Addresssin). Thus, to avoid regenerating the graph strucutre during trainig, COALA-GNN loads CSC format. Run the following commands in `example` directory to create CSC graph format.
```
cd examples
python create_csc_graph.py --path  $IGB_DATASET_PATH  --dataset_size medium  --data IGB 
python create_csc_graph.py --path  $OGB_DATASET_PATH   --data OGB 
```
`csc_indptr.npy`, `csc_indices.npy`, and `csc_edge_ids.npy` files should be generated in the `path` directoy.

### Artifact Execution
Users can run `sbatch_ssd_gnn_train.py` with srun to train GNN models with COALA-GNN. The following command is an example command to train GraphSage model with COALA-GNN for IGB-medium. `feat_cpu` command is to disable storage-based GNN traing.

```
srun --label python sbatch_ssd_gnn_train.py --path $IGB_DATA_PATH  --dataset_size medium  --data IGB --fan_out "5,5" --num_layers 2 --batch_size 1024 --cache_size 4096 --feat_cpu --model_type sage --distribution node_color --cache_backend nvshmem
```
### Scripts
The following scripts are used to run COALA-GNN with different configurationns. Users can run sbatch with these scripts. Please double check SBATCH parameters for the environment, such as account, partitions, or times.
- IGB_4GB_script.sh (running IGB dataset with 4GB cache size)
- IGB_16GB_script.sh (running IGB dataset with 16GB cache size)
- OGB_4GB_script.sh (running OGB dataset with 4GB cache size)
- OGB_16GB_script.sh (running OGB dataset with 16GB cache size)
- Cache_compare_script.sh (compare NVSHMEM-based cache, NCCL-based cache, and Isolated cache)
- Distribution_compare_script.sh (compare dynamic node distribuiton vs baseline striping)
- Pipeline_compare_script.sh (compare when pipeline is on and off)
  


