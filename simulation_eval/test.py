import torch
import cupy as cp

import BAM_Feature_Store
from BAM_Feature_Store import Emulate_SA


def ptr_to_tensor(device_ptr: int, nbytes: int, shape: tuple):
    mem = cp.cuda.UnownedMemory(device_ptr, nbytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, offset=0)
    arr = cp.ndarray(shape, dtype=cp.float32, memptr=memptr)
    return torch.as_tensor(arr, device="cuda"), memptr


shape = [3,3]
x = torch.ones([3,3], device="cuda:0", dtype=torch.float32)

b,memptr = ptr_to_tensor(x.data_ptr(), 9 * 4, [3,3])

# arr = cp.ndarray(shape, dtype=cp.float32, memptr=x.data_ptr())
# b = torch.as_tensor(arr, device="cuda")
print("b: ", b)

cur_Emul_SA = Emulate_SA()
cur_Emul_SA.write_data(x.data_ptr(), 9)

print("b: ", b)

a = b.to("cpu")
print("a: ", a)

print("sum: ", torch.sum(b))



# # Create a PyTorch tensor
# x = torch.ones([3,3], device="cuda:0")
# print("x ptr: ", x.data_ptr())
# # Convert to DLPack capsule
# dlpack_capsule = torch.utils.dlpack.to_dlpack(x)

# print(dlpack_capsule)


# b = torch.utils.dlpack.from_dlpack(dlpack_capsule)
# print(b)
# print("b ptr: ", b.data_ptr())

# c = torch.zeros([3,3], device="cuda:0")
# print("c ptr: ", c.data_ptr())
# b_host = b.to("cpu")
# print("b host: ", b_host)



# b.data_ptr = c.data_ptr
# print("b ptr: ", b.data_ptr())
# b_host = b.to("cpu")

# print("b host: ", b_host)
