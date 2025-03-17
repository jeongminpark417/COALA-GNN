import os
import torch.distributed as dist
import dgl
import Dist_Cache
# def main():
#     # Get rank and world size from environment
#     # rank = int(os.environ["RANK"])
#     # world_size = int(os.environ["WORLD_SIZE"])

#     rank = int(os.environ.get('SLURM_PROCID'))
#     world_size = int(os.environ.get('SLURM_NTASKS'))
#     print(f"Rank: {rank} Size:{world_size}")
#     # Initialize the process group
#     dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

#     # Your training code here
#     print(f"Rank {rank} initialized, World Size {world_size}")

os.environ['MASTER_ADDR'] = 'gpub031'
os.environ['MASTER_PORT'] = '12349'

# main()
print("hi")
rank = int(os.environ.get('SLURM_PROCID'))
world_size = int(os.environ.get('SLURM_NTASKS'))
print(f"Rank: {rank} Size:{world_size}")
dist.init_process_group(backend="gloo",  world_size=world_size, rank=rank)
print(f"Rank: {rank} done")
dist.barrier()

rlist1 = [0,1]
rlist2 = [2,3]

if (rank == 0 or rank == 1):
    rlist = [0,1]
else:
    rlist = [2,3]

print(f"Rank: {rank}  Node ID: {rlist}" )

local_gloo = dist.new_group(ranks=rlist, backend='gloo')
dist.barrier(local_gloo)

dist.barrier()

# slurm_node_id = int(os.environ.get('SLURM_NODEID'))
# print(f"Rank: {rank}  Node ID: {slurm_node_id} Ranks: {[int(slurm_node_id * 2), int(slurm_node_id*2 + 1)]}" )
# local_gloo = dist.new_group( ranks=[int(slurm_node_id * 2), int(slurm_node_id*2 + 1)], backend='gloo')
# dist.barrier(local_gloo)
# print(f"Test 3MPI rank: {rank}  ")

   

# NVSHMEM_Cache = Dist_Cache.NVSHMEM_Cache()
# NVSHMEM_Cache.init(0)
# nvshmem_world_size = NVSHMEM_Cache.get_world_size()
# mype_node = NVSHMEM_Cache.get_mype_node()
# nvshmem_rank = NVSHMEM_Cache.get_rank()
# nbytes = int(5 * 5 * 2 * 1024 * 4 * 1024)
# nbytes = 1024
# print("PI: ", NVSHMEM_Cache)
# nvshmem_ptr = NVSHMEM_Cache.allocate(nbytes) 
# print(f"NVShmem Size: {nvshmem_world_size} rank:{nvshmem_rank} PE_Node:{mype_node}")


# # all_mype_nodes = [None] * world_size
# # dist.all_gather_object(all_mype_nodes, mype_node)

# # # Identify ranks where my_pe_node == 0
# # ranks_with_node_0 = [i for i, node in enumerate(all_mype_nodes) if node == 0]
# # print(f"Rank: {rank} result: {ranks_with_node_0}")
# # Create Gloo process group for ranks where my_pe_node == 0
# gloo_group = dist.new_group( backend='gloo')
# gloo_world_size = dist.get_world_size(group=gloo_group)
# print(f"NVShmem Size: {nvshmem_world_size} rank:{nvshmem_rank}

#torch.distributed.new_group(backend='gloo')