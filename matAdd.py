import numpy as np
from numba import cuda

@cuda.jit
def mat_add(d_A, d_B, d_C):
    i, j = cuda.grid(2)
    if i < d_C.shape[0] and j < d_C.shape[1]:
        d_C[i, j] = d_A[i, j] + d_B[i, j]

# Matrix dimensions
n, m = 1024, 1024

# Initialize matrices on the host
h_A = np.random.rand(n, m).astype(np.float32)
h_B = np.random.rand(n, m).astype(np.float32)

# Allocate memory on the device and copy data from host to device
d_A = cuda.to_device(h_A)
d_B = cuda.to_device(h_B)
d_C = cuda.device_array((n, m), dtype=np.float32)

# Define thread and grid dimensions
threads_per_block = (16, 16)
blocks_per_grid = (int(np.ceil(n / threads_per_block[0])), int(np.ceil(m / threads_per_block[1])))

# Launch the kernel
mat_add[blocks_per_grid, threads_per_block](d_A, d_B, d_C)

# Copy result from device to host
h_C = d_C.copy_to_host()
print(h_C)
