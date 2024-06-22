import numpy as np
from numba import cuda

@cuda.jit
def mat_mul(d_A, d_B, d_C):
    i, j = cuda.grid(2)
    if i < d_C.shape[0] and j < d_C.shape[1]:
        temp = 0
        for k in range(d_A.shape[1]):
            temp += d_A[i, k] * d_B[k, j]
        d_C[i, j] = temp

n, m = 1024, 1024

h_A = np.random.rand(n, m).astype(np.float32)
h_B = np.random.rand(m, n).astype(np.float32)

d_A = cuda.to_device(h_A)
d_B = cuda.to_device(h_B)

d_C = cuda.device_array((n, n), dtype=np.float32)

threads_per_block = (16, 16)
blocks_per_grid = (int(np.ceil(n / threads_per_block[0])), int(np.ceil(n / threads_per_block[1])))

mat_mul[blocks_per_grid, threads_per_block](d_A, d_B, d_C)

h_C = d_C.copy_to_host()

print(h_C)
