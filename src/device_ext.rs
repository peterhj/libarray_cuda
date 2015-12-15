use device_memory::{DeviceBufferRef, DeviceBufferRefMut};
use ffi::*;

use cuda::runtime::{cuda_memset_async};
use libc::{c_int};

pub trait DeviceBytesExt {
  fn set_memory(&mut self, c: u8);
}

impl<'ctx> DeviceBytesExt for DeviceBufferRefMut<'ctx, u8> {
  fn set_memory(&mut self, c: u8) {
    unsafe { cuda_memset_async(
        self.as_mut_ptr(),
        c as i32,
        self.len(),
        &self.ctx.stream,
    ) }.unwrap();
  }
}

pub trait DeviceNumExt<T> {
  type Ref;

  fn print(&mut self);
  fn set_constant(&mut self, c: T);
  fn add(&mut self, other: &Self::Ref);
}

impl<'ctx> DeviceNumExt<i32> for DeviceBufferRefMut<'ctx, i32> {
  type Ref = DeviceBufferRef<'ctx, i32>;

  fn print(&mut self) {
    unsafe { array_cuda_map_print_i32(
        self.as_ptr(), self.len() as c_int,
        self.ctx.stream.ptr,
    ) };
  }

  fn set_constant(&mut self, c: i32) {
    unsafe { array_cuda_map_set_constant_i32(
        self.as_mut_ptr(), self.len() as c_int,
        c,
        self.ctx.stream.ptr,
    ) };
  }

  fn add(&mut self, other: &DeviceBufferRef<'ctx, i32>) {
    unsafe { array_cuda_map_add_i32(
        other.as_ptr(), other.len() as c_int,
        self.as_mut_ptr(),
        self.ctx.stream.ptr,
    ) };
  }
}

impl<'ctx> DeviceNumExt<f32> for DeviceBufferRefMut<'ctx, f32> {
  type Ref = DeviceBufferRef<'ctx, f32>;

  fn print(&mut self) {
    unimplemented!();
  }

  fn set_constant(&mut self, c: f32) {
    unsafe { array_cuda_map_set_constant_f32(
        self.as_mut_ptr(), self.len() as c_int,
        c,
        self.ctx.stream.ptr,
    ) };
  }

  fn add(&mut self, other: &DeviceBufferRef<'ctx, f32>) {
    unsafe { array_cuda_map_add_f32(
        other.as_ptr(), other.len() as c_int,
        self.as_mut_ptr(),
        self.ctx.stream.ptr,
    ) };
  }
}
