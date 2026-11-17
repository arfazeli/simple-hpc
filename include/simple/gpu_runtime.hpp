#pragma once
// One GPU source for AMD and NVIDIA. The kernels and host code are written in HIP; a CUDA build
// (SIMPLE_CUDA, compiled by nvcc) maps the HIP runtime names used in this repo to their CUDA twins.
// Kernel syntax (__global__, <<<>>>, __shared__, threadIdx, ...) is the same in both.
// ponytail: only the calls the .hip files use; add a line when one starts using another.
#ifdef SIMPLE_CUDA
#include <cuda_runtime.h>
#define hipDeviceProp_t cudaDeviceProp
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipError_t cudaError_t
#define hipEventCreate cudaEventCreate
#define hipEventElapsedTime cudaEventElapsedTime
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEvent_t cudaEvent_t
#define hipFree cudaFree
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipGetErrorString cudaGetErrorString
#define hipGetLastError cudaGetLastError
#define hipMalloc cudaMalloc
#define hipMemcpy cudaMemcpy
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemset cudaMemset
#define hipSuccess cudaSuccess
#else
#include <hip/hip_runtime.h>
#endif
