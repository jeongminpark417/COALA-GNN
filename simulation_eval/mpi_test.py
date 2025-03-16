import dgl
import Dist_Cache

NVSHMEM_Cache = Dist_Cache.NVSHMEM_Cache()
NVSHMEM_Cache.init(0)
world_size = NVSHMEM_Cache.get_world_size()
mype_node = NVSHMEM_Cache.get_mype_node()
rank = NVSHMEM_Cache.get_rank()
nbytes = int(5 * 5 * 2 * 1024 * 4 * 1024)
nbytes = 10224
print("PI: ", NVSHMEM_Cache)
nvshmem_ptr = NVSHMEM_Cache.allocate(nbytes) 
print(f"W Size: {world_size} rank:{rank} PE_Node:{mype_node}")

