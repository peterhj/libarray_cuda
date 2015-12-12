use cuda::ffi::runtime::{cudaStream_t};
use libc::{c_int};

#[link(name = "array_cuda_kernels", kind = "static")]
extern "C" {
  pub fn array_cuda_map_print_i32(src: *const i32, n: c_int, stream: cudaStream_t);
  pub fn array_cuda_map_set_constant_i32(src: *mut i32, n: c_int, c: i32, stream: cudaStream_t);
  pub fn array_cuda_map_add_i32(src: *const i32, n: c_int, dst: *mut i32, stream: cudaStream_t);
}
