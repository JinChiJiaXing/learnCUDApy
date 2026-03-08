from numba import cuda, float32
import numpy as np
import numba
import math
import time

TPB = 16

@numba.jit(nopython=True)
def matmul_cpu(A, B, C):   
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp


@cuda.jit
def matmul_gpu(A, B, C):
    i, j = cuda.grid(2)
    if i >= C.shape[0] or j >= C.shape[1]:
        return
    tmp = 0.
    for k in range(A.shape[1]):
        tmp += A[i, k] * B[k, j] # 多次访问GPU全局内存，性能较差
    C[i, j] = tmp


@cuda.jit
def matmul_shared_mem(A, B, C):
    sA = cuda.shared.array((TPB, TPB), dtype=float32)
    sB = cuda.shared.array((TPB, TPB), dtype=float32)
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    if x >= C.shape[0] or y >= C.shape[1]:
        return
    tmp = 0.
    for i in range(int(A.shape[1]/TPB)):
        sA[tx, ty] = A[x, ty + i*TPB]
        sB[tx, ty] = B[tx + i*TPB, y]
        cuda.syncthreads() # 同步线程，所有线程都需要把数据搬运完毕才能进行计算
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        cuda.syncthreads() # 同步线程
    C[x, y] = tmp

if __name__ == "__main__": 
    # 初始化矩阵
    A = np.random.random((TPB*500, TPB*200)).astype(np.float32)
    B = np.random.random((TPB*200, TPB*500)).astype(np.float32)
    C_cpu = np.full((A.shape[0], B.shape[1]), 0, dtype=np.float32)
    
    # CPU计算
    print("Running CPU matmul...")
    start_cpu = time.time()
    matmul_cpu(A, B, C_cpu)
    end_cpu = time.time()
    print(f"CPU matmul time: {end_cpu - start_cpu:.4f} seconds")

    # GPU内存初始化
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    C_global_mem = cuda.device_array((A.shape[0], B.shape[1]), dtype=np.float32)
    C_shared_mem = cuda.device_array((A.shape[0], B.shape[1]), dtype=np.float32)

    threadsperblock = (TPB, TPB)
    blockspergrid_x = int(math.ceil(C_cpu.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(C_cpu.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # GPU计算 (全局内存)
    print("Running GPU matmul...")
    start_gpu = time.time()
    matmul_gpu[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
    end_gpu = time.time()
    print(f"GPU matmul time (global memory): {end_gpu - start_gpu:.4f} seconds")
    C_global_gpu = C_global_mem.copy_to_host()

    # GPU计算（共享内存）
    print("Running GPU matmul with shared memory...")
    start_gpu_shared = time.time()
    matmul_shared_mem[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_shared_mem)
    end_gpu_shared = time.time()
    print(f"GPU matmul time (shared memory): {end_gpu_shared - start_gpu_shared:.4f} seconds")
    C_shared_gpu = C_shared_mem.copy_to_host()

