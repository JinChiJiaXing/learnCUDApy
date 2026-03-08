import cv2
import numpy as np
from numba import cuda
import time
import math

def process_cpu(img, dst):
    rows, cols, channels = img.shape
    for i in range(rows):
        for j in range(cols):
            for k in range(channels):
                dst[i, j, k] = 255 - img[i, j, k]

@cuda.jit
def process_gpu(img, channels):
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    for k in range(channels):
        if tx < img.shape[0] and ty < img.shape[1]:
            img[tx, ty, k] = 255 - img[tx, ty, k]

if __name__ == "__main__":
    print("Hello, World!")
    img = cv2.imread("./test_img.PNG")
    rows, cols, channels = img.shape
    dst_cpu = img.copy()
    dst_gpu = img.copy()

    start_cpu = time.time()
    process_cpu(img,dst_cpu)
    end_cpu = time.time()
    print(f"CPU processing time: {end_cpu - start_cpu:.4f} seconds")


    treadsperblock = (16, 16)
    blockspergrid_x = int(math.ceil(rows / treadsperblock[0]))
    blockspergrid_y = int(math.ceil(cols / treadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    cuda.synchronize()

    start_gpu = time.time()
    dImg = cuda.to_device(img)
    process_gpu[blockspergrid, treadsperblock](dImg, channels)
    end_gpu = time.time()
    dist_gpu = dImg.copy_to_host()
    print(f"GPU processing time: {end_gpu - start_gpu:.4f} seconds")

    cv2.imwrite("dst_cpu.jpg", dst_cpu)
    cv2.imwrite("dst_gpu.jpg", dist_gpu)