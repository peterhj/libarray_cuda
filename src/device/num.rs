use device::array::{DeviceArray2dView, DeviceArray2dViewMut};
use device::context::{DeviceCtxRef};
use device::memory::{DeviceBufferRef, DeviceBufferRefMut, RawDeviceBufferRef};
use ffi::*;

use array::{ArrayView, ArrayViewMut};

use cuda::runtime::{cuda_memset_async};
use libc::{c_int};

pub trait BytesExt {
  fn set_memory(&mut self, c: u8);
}

impl<'ctx> BytesExt for DeviceBufferRefMut<'ctx, u8> {
  fn set_memory(&mut self, c: u8) {
    unsafe { cuda_memset_async(
        self.as_mut_ptr(),
        c as i32,
        self.len(),
        &self.ctx.stream,
    ) }.unwrap();
  }
}

pub trait CastBytesExt<T> {
  type Ref;

  fn cast_bytes(&self, dst: &mut Self::Ref);
  fn cast_bytes_normalized(&self, dst: &mut Self::Ref);
}

impl<'ctx> CastBytesExt<f32> for DeviceBufferRef<'ctx, u8> {
  type Ref = DeviceBufferRefMut<'ctx, f32>;

  fn cast_bytes(&self, dst: &mut DeviceBufferRefMut<'ctx, f32>) {
    unsafe { array_cuda_map_cast_u8_to_f32_vec(
        self.as_ptr(), self.len() as i32,
        dst.as_mut_ptr(),
        self.ctx.stream.ptr,
    ) };
  }

  fn cast_bytes_normalized(&self, dst: &mut DeviceBufferRefMut<'ctx, f32>) {
    unsafe { array_cuda_map_cast_u8_to_f32_vec_norm(
        self.as_ptr(), self.len() as i32,
        dst.as_mut_ptr(),
        self.ctx.stream.ptr,
    ) };
  }
}

pub trait AsyncNumExt<T> {
  type Ctx;

  fn async_set_constant(&self, alpha: T, ctx: &Self::Ctx);
}

pub trait NumExt<T> {
  type Ref;

  fn print(&mut self);
  fn set_constant(&mut self, c: T);
  fn add(&mut self, other: &Self::Ref);
}

impl<'ctx> NumExt<i32> for DeviceBufferRefMut<'ctx, i32> {
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

impl<'ctx> AsyncNumExt<f32> for RawDeviceBufferRef<'ctx, f32> {
  //type Ref = DeviceBufferRef<'ctx, f32>;
  type Ctx = DeviceCtxRef<'ctx>;

  fn async_set_constant(&self, alpha: f32, ctx: &DeviceCtxRef<'ctx>) {
    unsafe { array_cuda_map_set_constant_f32(
        self.as_mut_ptr(), self.len() as c_int,
        alpha,
        ctx.stream.ptr,
    ) };
  }

  /*fn add(&mut self, other: &DeviceBufferRef<'ctx, i32>) {
    unsafe { array_cuda_map_add_i32(
        other.as_ptr(), other.len() as c_int,
        self.as_mut_ptr(),
        self.ctx.stream.ptr,
    ) };
  }*/
}

impl<'ctx> NumExt<f32> for DeviceBufferRefMut<'ctx, f32> {
  type Ref = DeviceBufferRef<'ctx, f32>;

  fn print(&mut self) {
    unsafe { array_cuda_map_print_f32(
        self.as_ptr(), self.len() as c_int,
        self.ctx.stream.ptr,
    ) };
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
        1.0, other.as_ptr(), other.len() as c_int,
        1.0, self.as_mut_ptr(),
        self.ctx.stream.ptr,
    ) };
  }
}

impl<'ctx> NumExt<f32> for DeviceArray2dViewMut<'ctx, f32> {
  type Ref = DeviceArray2dView<'ctx, f32>;

  fn print(&mut self) {
    unimplemented!();
  }

  fn set_constant(&mut self, c: f32) {
    unsafe { array_cuda_map_set_constant_f32(
        self.as_mut_ptr(), self.len() as c_int,
        c,
        self.data.ctx.stream.ptr,
    ) };
  }

  fn add(&mut self, other: &DeviceArray2dView<'ctx, f32>) {
    // TODO(20151218)
    unimplemented!();
  }
}
