//#![feature(arc_counts)]
#![feature(cell_extras)]
#![feature(optin_builtin_traits)]
#![feature(unique)]
//#![feature(zero_one)]

extern crate array_new;
extern crate cuda;
extern crate cuda_blas;
extern crate cuda_dnn;
extern crate cuda_rand;
extern crate cuda_sparse;

extern crate libc;
extern crate rand;
extern crate vec_map;

pub mod device;
pub mod ffi;
pub mod host_memory;
