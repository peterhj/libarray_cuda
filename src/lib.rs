#![feature(arc_counts)]
#![feature(optin_builtin_traits)]
#![feature(unique)]

//extern crate async_cuda;
extern crate cuda;
extern crate cuda_blas;
extern crate cuda_dnn;
extern crate cuda_rand;
extern crate cuda_sparse;

extern crate libc;
extern crate rand;
extern crate vec_map;

pub mod context;
pub mod device_comm;
pub mod device_ext;
pub mod device_linalg;
pub mod device_memory;
pub mod device_sync;
pub mod ffi;
pub mod host_memory;
