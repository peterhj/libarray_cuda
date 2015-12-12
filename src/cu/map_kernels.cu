#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>

__global__ void map_print_i32_kernel(
    const int32_t *src, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    printf("DEBUG: print: %d %d\n", i, src[i]);
  }
}

extern "C" void array_cuda_map_print_i32(
    const int32_t *src, int n,
    cudaStream_t stream)
{
  map_print_i32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      src, n);
}

__global__ void map_set_constant_i32_kernel(
    int32_t *src, int n,
    int32_t c)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    src[i] = c;
  }
}

extern "C" void array_cuda_map_set_constant_i32(
    int32_t *src, int n,
    int32_t c,
    cudaStream_t stream)
{
  map_set_constant_i32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      src, n, c);
}

__global__ void map_set_constant_f32_kernel(
    float *src, int n,
    float c)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    src[i] = c;
  }
}

extern "C" void array_cuda_map_set_constant_f32(
    float *src, int n,
    float c,
    cudaStream_t stream)
{
  map_set_constant_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      src, n, c);
}

__global__ void map_add_i32_kernel(
    const int32_t *src, int n,
    int32_t *dst)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    dst[i] = dst[i] + src[i];
  }
}

extern "C" void array_cuda_map_add_i32(
    const int32_t *src, int n,
    int32_t *dst,
    cudaStream_t stream)
{
  map_add_i32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      src, n, dst);
}

__global__ void map_add_f32_kernel(
    const float *src, int n,
    float *dst)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    dst[i] = dst[i] + src[i];
  }
}

extern "C" void array_cuda_map_add_f32(
    const float *src, int n,
    float *dst,
    cudaStream_t stream)
{
  map_add_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      src, n, dst);
}
