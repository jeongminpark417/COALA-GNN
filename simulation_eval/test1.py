import numpy


import cupy as cp

from DistEdgeDataLoader import Simulation_Loader

# Create two large random arrays
a = cp.random.rand(1000000)
b = cp.random.rand(1000000)

# Perform element-wise addition
c = a + b

# Perform element-wise multiplication
d = a * b

# Calculate the sum of all elements in the resulting array
sum_c = cp.sum(c)
sum_d = cp.sum(d)

print("Sum of elements in c:", sum_c)
print("Sum of elements in d:", sum_d)
