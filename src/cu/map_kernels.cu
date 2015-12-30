#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>

#define BANK_OFFSET(idx) ({ __typeof__ (idx) _idx = idx; ((_idx) + ((_idx) / 32)); })

__global__ void map_print_i32_kernel(
    const int32_t *src, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    printf("DEBUG: print: [%d] %d\n", i, src[i]);
  }
}

extern "C" void array_cuda_map_print_i32(
    const int32_t *src, int n,
    cudaStream_t stream)
{
  map_print_i32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      src, 32);
}

__global__ void map_print_f32_kernel(
    const float *src, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    printf("DEBUG: print: [%d] %g\n", i, src[i]);
  }
}

extern "C" void array_cuda_map_print_f32(
    const float *src, int n,
    cudaStream_t stream)
{
  map_print_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      src, 32);
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

__global__ void map_cast_u8_to_f32_v(
    const uint32_t *vsrc, int vn,
    float *dst, int n)
{
  int vi = threadIdx.x + blockIdx.x * blockDim.x;
  int i0 = 4 * vi;
  int i1 = i0 + 1;
  int i2 = i0 + 2;
  int i3 = i0 + 3;
  if (vi < vn) {
    uint32_t v = vsrc[vi];
    float x0 = (float)(v & 0xff);
    float x1 = (float)((v >> 8) & 0xff);
    float x2 = (float)((v >> 16) & 0xff);
    float x3 = (float)((v >> 24) & 0xff);
    if (i0 < n) {
      dst[i0] = x0;
    }
    if (i1 < n) {
      dst[i1] = x1;
    }
    if (i2 < n) {
      dst[i2] = x2;
    }
    if (i3 < n) {
      dst[i3] = x3;
    }
  }
}

__global__ void map_cast_u8_to_f32_vs(
    const uint32_t *vsrc, int vn,
    float *dst, int n)
{
  __shared__ float cache[4 * (1024 + 32)];
  int vi = threadIdx.x + blockIdx.x * blockDim.x;
  int i0 = threadIdx.x + 4 * blockIdx.x * blockDim.x;
  int i1 = i0 + 1024;
  int i2 = i0 + 2 * 1024;
  int i3 = i0 + 3 * 1024;
  if (vi < vn) {
    uint32_t v = vsrc[vi];
    cache[BANK_OFFSET(4 * threadIdx.x)]     = (float)(v & 0xff);
    cache[BANK_OFFSET(4 * threadIdx.x + 1)] = (float)((v >> 8) & 0xff);
    cache[BANK_OFFSET(4 * threadIdx.x + 2)] = (float)((v >> 16) & 0xff);
    cache[BANK_OFFSET(4 * threadIdx.x + 3)] = (float)((v >> 24) & 0xff);
    __syncthreads();
    if (i0 < n) {
      dst[i0] = cache[BANK_OFFSET(threadIdx.x)];
    }
    if (i1 < n) {
      dst[i1] = cache[BANK_OFFSET(threadIdx.x + 1024)];
    }
    if (i2 < n) {
      dst[i2] = cache[BANK_OFFSET(threadIdx.x + 2 * 1024)];
    }
    if (i3 < n) {
      dst[i3] = cache[BANK_OFFSET(threadIdx.x + 3 * 1024)];
    }
  }
}

extern "C" void array_cuda_map_cast_u8_to_f32_vec(
    const uint8_t *src, int n,
    float *dst,
    cudaStream_t stream)
{
  int vn = (n+3)/4;
  map_cast_u8_to_f32_v<<<(vn+1024-1)/1024, 1024, 0, stream>>>(
      (const uint32_t *)src, vn, dst, n);
}

__global__ void map_cast_u8_to_f32_v_n(
    const uint32_t *vsrc, int vn,
    float *dst, int n)
{
  int vi = threadIdx.x + blockIdx.x * blockDim.x;
  int i0 = 4 * vi;
  int i1 = i0 + 1;
  int i2 = i0 + 2;
  int i3 = i0 + 3;
  if (vi < vn) {
    uint32_t v = vsrc[vi];
    float x0 = (float)(v & 0xff) / 255.0f;
    float x1 = (float)((v >> 8) & 0xff) / 255.0f;
    float x2 = (float)((v >> 16) & 0xff) / 255.0f;
    float x3 = (float)((v >> 24) & 0xff) / 255.0f;
    if (i0 < n) {
      dst[i0] = x0;
    }
    if (i1 < n) {
      dst[i1] = x1;
    }
    if (i2 < n) {
      dst[i2] = x2;
    }
    if (i3 < n) {
      dst[i3] = x3;
    }
  }
}

__global__ void map_cast_u8_to_f32_vs_n(
    const uint32_t *vsrc, int vn,
    float *dst, int n)
{
  __shared__ float cache[4 * (1024 + 32)];
  int vi = threadIdx.x + blockIdx.x * blockDim.x;
  int i0 = threadIdx.x + 4 * blockIdx.x * blockDim.x;
  int i1 = i0 + 1024;
  int i2 = i0 + 2 * 1024;
  int i3 = i0 + 3 * 1024;
  if (vi < vn) {
    uint32_t v = vsrc[vi];
    cache[BANK_OFFSET(4 * threadIdx.x)]     = (float)(v & 0xff) / 255.0f;
    cache[BANK_OFFSET(4 * threadIdx.x + 1)] = (float)((v >> 8) & 0xff) / 255.0f;
    cache[BANK_OFFSET(4 * threadIdx.x + 2)] = (float)((v >> 16) & 0xff) / 255.0f;
    cache[BANK_OFFSET(4 * threadIdx.x + 3)] = (float)((v >> 24) & 0xff) / 255.0f;
    __syncthreads();
    if (i0 < n) {
      dst[i0] = cache[BANK_OFFSET(threadIdx.x)];
    }
    if (i1 < n) {
      dst[i1] = cache[BANK_OFFSET(threadIdx.x + 1024)];
    }
    if (i2 < n) {
      dst[i2] = cache[BANK_OFFSET(threadIdx.x + 2 * 1024)];
    }
    if (i3 < n) {
      dst[i3] = cache[BANK_OFFSET(threadIdx.x + 3 * 1024)];
    }
  }
}

extern "C" void array_cuda_map_cast_u8_to_f32_vec_norm(
    const uint8_t *src, int n,
    float *dst,
    cudaStream_t stream)
{
  int vn = (n+3)/4;
  map_cast_u8_to_f32_v_n<<<(vn+1024-1)/1024, 1024, 0, stream>>>(
      (const uint32_t *)src, vn, dst, n);
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
