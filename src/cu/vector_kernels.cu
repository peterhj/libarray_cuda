#include <cuda_runtime_api.h>
// FIXME(20160123): commentng out for cuda 7.0.
//#include <cuda_fp16.h>

#include <stdint.h>

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
