use device::context::{DeviceCtxRef};
use device::memory::{DeviceZeroExt, DeviceBuffer, DeviceBufferRef, DeviceBufferRefMut};

use array_new::{
  Shape, Array, AsyncArray, ArrayView, ArrayViewMut,
  Array2dView, Array2dViewMut,
  Array3dView, Array3dViewMut,
};
use cuda::runtime::{
  CudaMemcpyKind,
  cuda_memcpy_async,
};

use std::mem::{size_of};

pub struct DeviceArray2d<T> where T: Copy {
  data:     DeviceBuffer<T>,
  bound:    (usize, usize),
  stride:   usize,
}

impl<'ctx, 'a, T> AsyncArray<'ctx, 'a, T, (usize, usize)> for DeviceArray2d<T>
where 'ctx: 'a, T: 'a + Copy {
  type Ctx = DeviceCtxRef<'ctx>;
  type View = DeviceArray2dView<'a, T>;
  type ViewMut = DeviceArray2dViewMut<'a, T>;

  fn as_view(&'a mut self, ctx: &'a DeviceCtxRef<'ctx>) -> DeviceArray2dView<'a, T> {
    DeviceArray2dView{
      data:     self.data.as_ref(ctx),
      bound:    self.bound,
      stride:   self.stride,
    }
  }

  fn as_view_mut(&'a mut self, ctx: &'a DeviceCtxRef<'ctx>) -> DeviceArray2dViewMut<'a, T> {
    DeviceArray2dViewMut{
      data:     self.data.as_ref_mut(ctx),
      bound:    self.bound,
      stride:   self.stride,
    }
  }
}

impl<T> DeviceArray2d<T> where T: Copy {
  pub unsafe fn new(bound: (usize, usize), ctx: &DeviceCtxRef) -> DeviceArray2d<T> {
    DeviceArray2d{
      data:     DeviceBuffer::new(bound.len(), ctx),
      bound:    bound,
      stride:   bound.to_least_stride(),
    }
  }
}

impl DeviceArray2d<u8> {
  pub fn zeros(bound: (usize, usize), ctx: &DeviceCtxRef) -> DeviceArray2d<u8> {
    DeviceArray2d{
      data:     DeviceBuffer::<u8>::zeros(bound.len(), ctx),
      bound:    bound,
      stride:   bound.to_least_stride(),
    }
  }
}

impl DeviceArray2d<i32> {
  pub fn zeros(bound: (usize, usize), ctx: &DeviceCtxRef) -> DeviceArray2d<i32> {
    DeviceArray2d{
      data:     DeviceBuffer::<i32>::zeros(bound.len(), ctx),
      bound:    bound,
      stride:   bound.to_least_stride(),
    }
  }
}

impl DeviceArray2d<f32> {
  pub fn zeros(bound: (usize, usize), ctx: &DeviceCtxRef) -> DeviceArray2d<f32> {
    DeviceArray2d{
      data:     DeviceBuffer::<f32>::zeros(bound.len(), ctx),
      bound:    bound,
      stride:   bound.to_least_stride(),
    }
  }

  pub fn len(&self) -> usize {
    self.bound.len()
  }
}

pub struct DeviceArray2dView<'a, T> where T: 'a + Copy {
  pub data:     DeviceBufferRef<'a, T>,
  // FIXME(20160201): bound and stride should not be public, only pub because of
  // .into_2d_view method of DeviceBufRef.
  pub bound:    (usize, usize),
  pub stride:   usize,
}

impl<'a, T> ArrayView<'a, T, (usize, usize)> for DeviceArray2dView<'a, T> where T: 'a + Copy {
  fn bound(&self) -> (usize, usize) {
    self.bound
  }

  fn stride(&self) -> usize {
    self.stride
  }

  fn len(&self) -> usize {
    self.bound.len()
  }

  unsafe fn as_ptr(&self) -> *const T {
    self.data.as_ptr()
  }

  fn view(self, lo: (usize, usize), hi: (usize, usize)) -> DeviceArray2dView<'a, T> {
    // TODO(20151214)
    unimplemented!();
  }
}

impl<'a, T> DeviceArray2dView<'a, T> where T: 'a + Copy {
  pub fn send(&self, dst: &mut DeviceArray2dViewMut<'a, T>) {
    assert_eq!(self.bound, dst.bound);
    if self.stride == dst.stride {
      self.data.send(&mut dst.data);
    } else {
      // TODO(20151214)
      unimplemented!();
    }
  }
}

impl<'a, T> DeviceArray2dView<'a, T> where T: 'a + Copy {
  pub fn sync_store(&self, dst: &mut Array2dViewMut<'a, T>) {
    assert_eq!(self.bound, dst.bound());
    if self.bound == dst.bound() {
      unsafe { cuda_memcpy_async(
          dst.as_mut_ptr() as *mut u8,
          self.as_ptr() as *const u8,
          self.bound.len() * size_of::<T>(),
          CudaMemcpyKind::DeviceToHost,
          &self.data.ctx.stream,
      ) }.unwrap();
      self.data.ctx.blocking_sync();
    } else {
      unimplemented!();
    }
  }
}

pub struct DeviceArray2dViewMut<'a, T> where T: 'a + Copy {
  pub data:     DeviceBufferRefMut<'a, T>,
  // FIXME(20160312): bound and stride should not be public, only pub because of
  // .into_2d_view_mut method of DeviceBufRefMut.
  pub bound:    (usize, usize),
  pub stride:   usize,
}

impl<'a, T> ArrayViewMut<'a, T, (usize, usize)> for DeviceArray2dViewMut<'a, T> where T: 'a + Copy {
  fn bound(&self) -> (usize, usize) {
    self.bound
  }

  fn stride(&self) -> usize {
    self.stride
  }

  fn len(&self) -> usize {
    self.bound.len()
  }

  unsafe fn as_ptr(&self) -> *const T {
    self.data.as_ptr()
  }

  unsafe fn as_mut_ptr(&mut self) -> *mut T {
    self.data.as_mut_ptr()
  }

  fn view_mut(self, lo: (usize, usize), hi: (usize, usize)) -> DeviceArray2dViewMut<'a, T> {
    // TODO(20151214)
    unimplemented!();
  }
}

impl<'a, T> DeviceArray2dViewMut<'a, T> where T: 'a + Copy {
  pub fn sync_load(&mut self, src: &Array2dView<'a, T>) {
    assert_eq!(self.bound, src.bound());
    if self.bound == src.bound() {
      unsafe { cuda_memcpy_async(
          self.as_mut_ptr() as *mut u8,
          src.as_ptr() as *const u8,
          self.bound.len() * size_of::<T>(),
          CudaMemcpyKind::HostToDevice,
          &self.data.ctx.stream,
      ) }.unwrap();
      self.data.ctx.blocking_sync();
    } else {
      unimplemented!();
    }
  }
}

pub struct DeviceArray3d<T> where T: Copy {
  data:     DeviceBuffer<T>,
  bound:    (usize, usize, usize),
  stride:   (usize, usize),
}

// TODO(20151214): 3d array impls.

pub struct DeviceArray3dView<'a, T> where T: 'a + Copy {
  pub data:     DeviceBufferRef<'a, T>,
  // FIXME(20160201): bound and stride should not be public, only pub because of
  // .into_3d_view method of DeviceBufRef.
  pub bound:    (usize, usize, usize),
  pub stride:   (usize, usize),
}

impl<'a, T> ArrayView<'a, T, (usize, usize, usize)> for DeviceArray3dView<'a, T> where T: 'a + Copy {
  fn bound(&self) -> (usize, usize, usize) {
    self.bound
  }

  fn stride(&self) -> (usize, usize) {
    self.stride
  }

  fn len(&self) -> usize {
    self.bound.len()
  }

  unsafe fn as_ptr(&self) -> *const T {
    self.data.as_ptr()
  }

  fn view(self, lo: (usize, usize, usize), hi: (usize, usize, usize)) -> DeviceArray3dView<'a, T> {
    // TODO(20151214)
    unimplemented!();
  }
}

impl<'a, T> DeviceArray3dView<'a, T> where T: 'a + Copy {
  pub fn send(&self, dst: &mut DeviceArray3dViewMut<'a, T>) {
    assert_eq!(self.bound, dst.bound);
    if self.stride == dst.stride {
      self.data.send(&mut dst.data);
    } else {
      // TODO(20151214)
      unimplemented!();
    }
  }
}

impl<'a, T> DeviceArray3dView<'a, T> where T: 'a + Copy {
  /*pub fn sync_store(&self, dst: &mut Array3dViewMut<'a, T>) {
    assert_eq!(self.bound, dst.bound());
    if self.bound == dst.bound() {
      unsafe { cuda_memcpy_async(
          dst.as_mut_ptr() as *mut u8,
          self.as_ptr() as *const u8,
          self.bound.len() * size_of::<T>(),
          CudaMemcpyKind::DeviceToHost,
          &self.data.ctx.stream,
      ) }.unwrap();
      self.data.ctx.blocking_sync();
    } else {
      unimplemented!();
    }
  }*/
}

pub struct DeviceArray3dViewMut<'a, T> where T: 'a + Copy {
  pub data:     DeviceBufferRefMut<'a, T>,
  // FIXME(20160312): bound and stride should not be public, only pub because of
  // .into_3d_view_mut method of DeviceBufRefMut.
  pub bound:    (usize, usize, usize),
  pub stride:   (usize, usize),
}

impl<'a, T> ArrayViewMut<'a, T, (usize, usize, usize)> for DeviceArray3dViewMut<'a, T> where T: 'a + Copy {
  fn bound(&self) -> (usize, usize, usize) {
    self.bound
  }

  fn stride(&self) -> (usize, usize) {
    self.stride
  }

  fn len(&self) -> usize {
    self.bound.len()
  }

  unsafe fn as_ptr(&self) -> *const T {
    self.data.as_ptr()
  }

  unsafe fn as_mut_ptr(&mut self) -> *mut T {
    self.data.as_mut_ptr()
  }

  fn view_mut(self, lo: (usize, usize, usize), hi: (usize, usize, usize)) -> DeviceArray3dViewMut<'a, T> {
    // TODO(20151214)
    unimplemented!();
  }
}

impl<'a, T> DeviceArray3dViewMut<'a, T> where T: 'a + Copy {
  /*pub fn sync_load(&mut self, src: &Array3dView<'a, T>) {
    assert_eq!(self.bound, src.bound());
    if self.bound == src.bound() {
      unsafe { cuda_memcpy_async(
          self.as_mut_ptr() as *mut u8,
          src.as_ptr() as *const u8,
          self.bound.len() * size_of::<T>(),
          CudaMemcpyKind::HostToDevice,
          &self.data.ctx.stream,
      ) }.unwrap();
      self.data.ctx.blocking_sync();
    } else {
      unimplemented!();
    }
  }*/
}

pub fn test_device_array(
    array1: &mut DeviceArray2d<f32>,
    array2: &mut DeviceArray2d<f32>,
    ctx: &DeviceCtxRef)
{
  let view1 = array1.as_view(ctx);
  let mut view2 = array2.as_view_mut(ctx);
  view1.send(&mut view2);
}
