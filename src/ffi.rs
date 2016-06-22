use cuda::ffi::runtime::{cudaStream_t};
use float::stub::{f16_stub};
use libc::{c_int};

#[link(name = "array_cuda_kernels", kind = "static")]
extern "C" {
  pub fn array_cuda_map_print_i32(src: *const i32, n: c_int, stream: cudaStream_t);
  pub fn array_cuda_map_print_f32(src: *const f32, n: c_int, stream: cudaStream_t);
  pub fn array_cuda_map_set_constant_i32(src: *mut i32, n: c_int, c: i32, stream: cudaStream_t);
  pub fn array_cuda_map_set_constant_f32(src: *mut f32, n: c_int, c: f32, stream: cudaStream_t);
  pub fn array_cuda_map_cast_u8_to_f32_vec(src: *const u8, n: c_int, dst: *mut f32, stream: cudaStream_t);
  pub fn array_cuda_map_cast_u8_to_f32_vec_norm(src: *const u8, n: c_int, dst: *mut f32, stream: cudaStream_t);
  pub fn array_cuda_map_cast_f16_to_f32(src: *const f16_stub, n: c_int, dst: *mut f32, stream: cudaStream_t);
  pub fn array_cuda_map_cast_f32_to_f16(src: *const f32, n: c_int, dst: *mut f16_stub, stream: cudaStream_t);
  pub fn array_cuda_map_add_i32(src: *const i32, n: c_int, dst: *mut i32, stream: cudaStream_t);
  pub fn array_cuda_map_add_f32(alpha: f32, src: *const f32, n: c_int, beta: f32, dst: *mut f32, stream: cudaStream_t);
  pub fn array_cuda_map_add_f16_as_f32(src: *const f16_stub, n: c_int, dst: *mut f16_stub, stream: cudaStream_t);

  pub fn array_cuda_vector_scale_f32(dst: *mut f32, dim: c_int, alpha: f32, stream: cudaStream_t);
  pub fn array_cuda_vector_exp_f32(xs: *mut f32, dim: c_int, stream: cudaStream_t);
  pub fn array_cuda_vector_add_f32(src: *const f32, dim: c_int, alpha: f32, beta: f32, dst: *mut f32, stream: cudaStream_t);
  pub fn array_cuda_vector_elemwise_mult_f32(xs: *const f32, len: c_int, ys: *mut f32, stream: cudaStream_t);
  pub fn array_cuda_vector_elemwise_div_f32(xs: *const f32, len: c_int, ys: *mut f32, stream: cudaStream_t);
}
