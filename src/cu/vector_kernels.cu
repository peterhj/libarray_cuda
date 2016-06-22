#include <cuda_runtime_api.h>
// FIXME(20160123): commentng out for cuda 7.0.
//#include <cuda_fp16.h>

#include <stdint.h>

__global__ void vector_scale_f32_kernel(
    float *dst,
    int dim,
    float alpha)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float y = alpha * dst[idx];
    dst[idx] = y;
  }
}

extern "C" void array_cuda_vector_scale_f32(
    float *dst,
    int dim,
    float alpha,
    cudaStream_t stream)
{
  vector_scale_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dst, dim, alpha);
}

__global__ void vector_exp_f32_kernel(
    float *xs,
    int dim)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x = expf(xs[idx]);
    xs[idx] = x;
  }
}

extern "C" void array_cuda_vector_exp_f32(
    float *xs,
    int dim,
    cudaStream_t stream)
{
  vector_exp_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      xs, dim);
}

__global__ void vector_add_f32_kernel(
    const float *src,
    int dim,
    float alpha,
    float beta,
    float *dst)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float y = alpha * src[idx] + beta * dst[idx];
    dst[idx] = y;
  }
}

extern "C" void array_cuda_vector_add_f32(
    const float *src,
    int dim,
    float alpha,
    float beta,
    float *dst,
    cudaStream_t stream)
{
  vector_add_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      src, dim, alpha, beta, dst);
}

__global__ void vector_elemwise_mult_f32_kernel(
    const float *xs,
    int len,
    float *ys)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    float y = xs[idx] * ys[idx];
    ys[idx] = y;
  }
}

extern "C" void array_cuda_vector_elemwise_mult_f32(
    const float *xs,
    int len,
    float *ys,
    cudaStream_t stream)
{
  vector_elemwise_mult_f32_kernel<<<(len+1024-1)/1024, 1024, 0, stream>>>(
      xs, len, ys);
}

__global__ void vector_elemwise_div_f32_kernel(
    const float *xs,
    int len,
    float *ys)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    float y = ys[idx] / xs[idx];
    ys[idx] = y;
  }
}

extern "C" void array_cuda_vector_elemwise_div_f32(
    const float *xs,
    int len,
    float *ys,
    cudaStream_t stream)
{
  vector_elemwise_div_f32_kernel<<<(len+1024-1)/1024, 1024, 0, stream>>>(
      xs, len, ys);
}
