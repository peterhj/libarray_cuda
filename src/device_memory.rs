use context::{DeviceCtxRef};
use device_ext::{DeviceBytesExt, DeviceNumExt};
use host_memory::{HostBufferRef};

use cuda::ffi::runtime::{
  cudaError,
  cudaFree,
  cudaMalloc
};
use cuda::runtime::{
  CudaEvent, CudaEventStatus, CudaMemcpyKind,
  cuda_memcpy_async, cuda_memcpy_peer_async,
};

use libc::{c_void};
use std::marker::{PhantomData};
use std::mem::{size_of};
use std::ptr::{null_mut};
use std::rc::{Rc};
use std::sync::{Arc, Mutex, MutexGuard};

const WARP_SIZE: usize = 128;

pub trait DeviceStorage<T> where T: Copy {
  type Ref:     DeviceStorageRef<T>;
  type RefMut:  DeviceStorageRefMut<T>;

  fn borrow<'ctx>(&mut self, ctx: &'ctx DeviceCtxRef<'ctx>) -> Self::Ref;
  fn borrow_mut<'ctx>(&mut self, ctx: &'ctx DeviceCtxRef<'ctx>) -> Self::RefMut;
}

pub trait DeviceStorageRef<T> where T: Copy {
  unsafe fn as_ptr(&self) -> *const T;
}

pub trait DeviceStorageRefMut<T> where T: Copy {
  unsafe fn as_ptr(&self) -> *const T;
  unsafe fn as_mut_ptr(&mut self) -> *mut T;
}

pub struct DeviceBuffer<T> where T: Copy {
  dev_sync: Rc<CudaEvent>,
  dev_idx:  usize,
  dptr: *mut T,
  len:  usize,
  size: usize,
}

impl<T> Drop for DeviceBuffer<T> where T: Copy {
  fn drop(&mut self) {
    match unsafe { cudaFree(self.dptr as *mut c_void) } {
      cudaError::Success => {}
      e => {
        panic!("failed to free device memory: {:?}", e);
      }
    }
  }
}

impl<T> DeviceBuffer<T> where T: Copy {
  pub unsafe fn new(len: usize, ctx: &DeviceCtxRef) -> DeviceBuffer<T> {
    let min_size = len * size_of::<T>();
    let size = (min_size + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    let mut dptr: *mut c_void = null_mut();
    match unsafe { cudaMalloc(&mut dptr as *mut *mut c_void, size) } {
      cudaError::Success => {}
      e => {
        panic!("failed to allocate DeviceBuffer: {:?}", e);
      }
    }
    DeviceBuffer{
      dev_sync: Rc::new(CudaEvent::create_with_flags(0x02).unwrap()),
      dev_idx:  ctx.device(),
      dptr: dptr as *mut T,
      len:  len,
      size: size,
    }
  }

  /*pub unsafe fn as_ptr(&self) -> *const T {
    self.dptr as *const T
  }

  pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
    self.dptr
  }*/

  pub fn borrow<'ctx>(&mut self, ctx: &'ctx DeviceCtxRef) -> DeviceBufferRef<'ctx, T> {
    ctx.stream.wait_event(&self.dev_sync).unwrap();
    DeviceBufferRef{
      ctx:  ctx,
      dev_sync: self.dev_sync.clone(),
      dev_idx:  self.dev_idx,
      dptr: self.dptr as *const T,
      len:  self.len,
      size: self.size,
    }
  }

  pub fn borrow_mut<'ctx>(&mut self, ctx: &'ctx DeviceCtxRef) -> DeviceBufferRefMut<'ctx, T> {
    ctx.stream.wait_event(&self.dev_sync).unwrap();
    DeviceBufferRefMut{
      ctx:  ctx,
      dev_sync: self.dev_sync.clone(),
      dev_idx:  self.dev_idx,
      dptr: self.dptr,
      len:  self.len,
      size: self.size,
    }
  }
}

impl DeviceBuffer<u8> {
  pub fn zeros(len: usize, ctx: &DeviceCtxRef) -> DeviceBuffer<u8> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.borrow_mut(ctx);
      buf_ref.set_memory(0);
    }
    buf
  }
}

impl DeviceBuffer<i32> {
  pub fn zeros(len: usize, ctx: &DeviceCtxRef) -> DeviceBuffer<i32> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.borrow_mut(ctx);
      buf_ref.set_constant(0);
    }
    buf
  }
}

impl DeviceBuffer<f32> {
  pub fn zeros(len: usize, ctx: &DeviceCtxRef) -> DeviceBuffer<f32> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.borrow_mut(ctx);
      buf_ref.set_constant(0.0);
    }
    buf
  }
}

pub struct DeviceBufferRef<'ctx, T> where T: 'ctx + Copy {
  pub ctx:  &'ctx DeviceCtxRef<'ctx>,
  dev_sync: Rc<CudaEvent>,
  dev_idx:  usize,
  dptr: *const T,
  len:  usize,
  size: usize,
}

impl<'ctx, T> Drop for DeviceBufferRef<'ctx, T> where T: 'ctx + Copy {
  fn drop(&mut self) {
    self.dev_sync.record(&self.ctx.stream).unwrap();
  }
}

impl<'ctx, T> DeviceBufferRef<'ctx, T> where T: 'ctx + Copy {
  pub fn len(&self) -> usize {
    self.len
  }

  pub unsafe fn as_ptr(&self) -> *const T {
    self.dptr as *const T
  }

  pub fn send(&self, other: &mut DeviceBufferRefMut<'ctx, T>) {
    assert_eq!(self.len, other.len);
    assert_eq!(self.size, other.size);
    if self.dev_idx == other.dev_idx {
      unsafe { cuda_memcpy_async(
          other.as_mut_ptr() as *mut u8,
          self.as_ptr() as *const u8,
          self.size,
          CudaMemcpyKind::DeviceToDevice,
          &self.ctx.stream,
      ) }.unwrap();
    } else {
      // TODO(20151211)
      unimplemented!();
    }
  }

  pub fn raw_send<'a>(&self, other: &RawDeviceBuffer<T>, ctx: &DeviceCtxRef<'a>) {
    assert_eq!(self.len, other.len);
    assert_eq!(self.size, other.size);
    if self.dev_idx == other.dev_idx {
      unsafe { cuda_memcpy_async(
          other.as_mut_ptr() as *mut u8,
          self.as_ptr() as *const u8,
          self.size,
          CudaMemcpyKind::DeviceToDevice,
          &ctx.stream,
      ) }.unwrap();
    } else {
      // TODO(20151211)
      unimplemented!();
    }
  }

  pub fn sync_store(&self, host_buf: &mut [T]) {
    assert_eq!(self.len, host_buf.len());
    unsafe { cuda_memcpy_async(
        host_buf.as_mut_ptr() as *mut u8,
        self.as_ptr() as *const u8,
        self.len * size_of::<T>(),
        CudaMemcpyKind::DeviceToHost,
        &self.ctx.stream,
    ) }.unwrap();
    self.ctx.blocking_sync();
  }
}

pub struct DeviceBufferRefMut<'ctx, T> where T: 'ctx + Copy {
  pub ctx:  &'ctx DeviceCtxRef<'ctx>,
  dev_sync: Rc<CudaEvent>,
  dev_idx:  usize,
  dptr: *mut T,
  len:  usize,
  size: usize,
}

impl<'ctx, T> Drop for DeviceBufferRefMut<'ctx, T> where T: 'ctx + Copy {
  fn drop(&mut self) {
    self.dev_sync.record(&self.ctx.stream).unwrap();
  }
}

impl<'ctx, T> DeviceBufferRefMut<'ctx, T> where T: 'ctx + Copy {
  pub fn len(&self) -> usize {
    self.len
  }

  pub unsafe fn as_ptr(&self) -> *const T {
    self.dptr as *const T
  }

  pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
    self.dptr
  }

  pub fn recv(&mut self, other: &DeviceBufferRef<'ctx, T>) {
    assert_eq!(self.len, other.len);
    if self.dev_idx == other.dev_idx {
      unsafe { cuda_memcpy_async(
          self.as_mut_ptr() as *mut u8,
          other.as_ptr() as *const u8,
          self.size,
          CudaMemcpyKind::DeviceToDevice,
          &self.ctx.stream,
      ) }.unwrap();
    } else {
      // TODO(20151211)
      unimplemented!();
    }
  }

  pub fn sync_store(&self, host_buf: &mut [T]) {
    assert_eq!(self.len, host_buf.len());
    unsafe { cuda_memcpy_async(
        host_buf.as_mut_ptr() as *mut u8,
        self.as_ptr() as *const u8,
        self.len * size_of::<T>(),
        CudaMemcpyKind::DeviceToHost,
        &self.ctx.stream,
    ) }.unwrap();
    self.ctx.blocking_sync();
  }

  pub fn sync_load(&mut self, host_buf: &[T]) {
    assert_eq!(self.len, host_buf.len());
    unsafe { cuda_memcpy_async(
        self.as_mut_ptr() as *mut u8,
        host_buf.as_ptr() as *const u8,
        self.len * size_of::<T>(),
        CudaMemcpyKind::HostToDevice,
        &self.ctx.stream,
    ) }.unwrap();
    self.ctx.blocking_sync();
  }

  pub fn load(&mut self, host_buf: &HostBufferRef<T>) {
    assert_eq!(self.len, host_buf.len());
    unsafe { cuda_memcpy_async(
        self.as_mut_ptr() as *mut u8,
        host_buf.as_ptr() as *const u8,
        self.len * size_of::<T>(),
        CudaMemcpyKind::HostToDevice,
        &self.ctx.stream,
    ) }.unwrap();
  }
}

pub struct RawDeviceBuffer<T> where T: Copy {
  dev_idx:  usize,
  dptr: *mut T,
  len:  usize,
  size: usize,
}

unsafe impl<T> Send for RawDeviceBuffer<T> where T: Copy {}
unsafe impl<T> Sync for RawDeviceBuffer<T> where T: Copy {}

impl<T> Drop for RawDeviceBuffer<T> where T: Copy {
  fn drop(&mut self) {
    match unsafe { cudaFree(self.dptr as *mut c_void) } {
      cudaError::Success => {}
      e => {
        panic!("failed to free device memory: {:?}", e);
      }
    }
  }
}

impl<T> RawDeviceBuffer<T> where T: Copy {
  pub unsafe fn new(len: usize, ctx: &DeviceCtxRef) -> RawDeviceBuffer<T> {
    let min_size = len * size_of::<T>();
    let size = (min_size + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    let mut dptr: *mut c_void = null_mut();
    match unsafe { cudaMalloc(&mut dptr as *mut *mut c_void, size) } {
      cudaError::Success => {}
      e => {
        panic!("failed to allocate DeviceBuffer: {:?}", e);
      }
    }
    RawDeviceBuffer{
      dev_idx:  ctx.device(),
      dptr: dptr as *mut T,
      len:  len,
      size: size,
    }
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub unsafe fn as_ptr(&self) -> *const T {
    self.dptr as *const T
  }

  pub unsafe fn as_mut_ptr(&self) -> *mut T {
    self.dptr
  }

  pub fn send<'ctx>(&self, other: &RawDeviceBuffer<T>, ctx: &DeviceCtxRef<'ctx>) {
    assert_eq!(self.len, other.len);
    assert_eq!(self.size, other.size);
    if self.dev_idx == other.dev_idx {
      /*unsafe { cuda_memcpy_async(
          other.as_mut_ptr() as *mut u8,
          self.as_ptr() as *const u8,
          self.size,
          CudaMemcpyKind::DeviceToDevice,
          &self.ctx.stream,
      ) }.unwrap();*/
      // TODO(20151212)
      unimplemented!();
    } else {
      unsafe { cuda_memcpy_peer_async(
          other.as_mut_ptr() as *mut u8, other.dev_idx,
          self.as_ptr() as *const u8, self.dev_idx,
          self.size,
          &ctx.stream,
      ) };
    }
  }
}

/*struct InnerSyncDeviceBuffer<T> {
  dev_sync: Option<Arc<CudaEvent>>,
  dev_idx:  usize,
  dptr: *mut T,
  len:  usize,
  size: usize,
}

unsafe impl<T> Send for InnerSyncDeviceBuffer<T> where T: Copy {}

pub struct SyncDeviceBuffer<T> where T: Copy {
  inner:    Arc<Mutex<InnerSyncDeviceBuffer<T>>>,
}

impl<T> Drop for SyncDeviceBuffer<T> where T: Copy {
  fn drop(&mut self) {
    let mut inner = self.inner.lock().unwrap();
    if let Some(dev_sync) = inner.dev_sync.take() {
      dev_sync.synchronize().unwrap();
    }
  }
}

impl<T> Clone for SyncDeviceBuffer<T> where T: Copy {
  fn clone(&self) -> SyncDeviceBuffer<T> {
    SyncDeviceBuffer{inner: self.inner.clone()}
  }
}

impl<T> SyncDeviceBuffer<T> where T: Copy {
  pub fn new(len: usize, ctx: &DeviceCtxRef) -> SyncDeviceBuffer<T> {
    let device = ctx.device();
    let min_size = len * size_of::<T>();
    let size = (min_size + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    let mut dptr: *mut c_void = null_mut();
    match unsafe { cudaMalloc(&mut dptr as *mut *mut c_void, size) } {
      cudaError::Success => {}
      e => {
        panic!("failed to allocate SyncDeviceBuffer: {:?}", e);
      }
    }
    SyncDeviceBuffer{
      inner:    Arc::new(Mutex::new(InnerSyncDeviceBuffer{
        dev_sync: None,
        dev_idx:  device,
        dptr: dptr as *mut T,
        len:  len,
        size: size,
      })),
    }
  }

  pub fn borrow<'ctx>(&'ctx mut self, ctx: &'ctx DeviceCtxRef<'ctx>) -> (SyncDeviceBufferRef<'ctx, T>, SyncGuard) {
    let mut inner = self.inner.lock().unwrap();
    if let Some(dev_sync) = inner.dev_sync.take() {
      ctx.stream.wait_event(&dev_sync).unwrap();
    }
    let guard = SyncGuard::new();
    (SyncDeviceBufferRef{
      ctx:      ctx,
      dev_sync: guard.dev_sync.clone(),
      inner:    Some(inner),
    }, guard)
  }

  pub fn borrow_mut<'ctx>(&'ctx mut self, ctx: &'ctx DeviceCtxRef<'ctx>) -> (SyncDeviceBufferRefMut<'ctx, T>, SyncGuard) {
    let mut inner = self.inner.lock().unwrap();
    if let Some(dev_sync) = inner.dev_sync.take() {
      ctx.stream.wait_event(&dev_sync).unwrap();
    }
    let guard = SyncGuard::new();
    (SyncDeviceBufferRefMut{
      ctx:      ctx,
      dev_sync: guard.dev_sync.clone(),
      inner:    Some(inner),
    }, guard)
  }
}

pub struct SyncDeviceBufferRef<'ctx, T> where T: 'ctx + Copy {
  pub ctx:  &'ctx DeviceCtxRef<'ctx>,
  dev_sync: Arc<CudaEvent>,
  inner:    Option<MutexGuard<'ctx, InnerSyncDeviceBuffer<T>>>,
}

impl<'ctx, T> Drop for SyncDeviceBufferRef<'ctx, T> where T: 'ctx + Copy {
  fn drop(&mut self) {
    self.dev_sync.record(&self.ctx.stream).unwrap();
    let mut inner = self.inner.take().unwrap();
    assert!(inner.dev_sync.is_none());
    inner.dev_sync = Some(self.dev_sync.clone());
  }
}

impl<'ctx, T> SyncDeviceBufferRef<'ctx, T> where T: 'ctx + Copy {
  pub fn join(self) {
  }

  pub fn len(&self) -> usize {
    self.inner.as_ref().unwrap().len
  }

  pub unsafe fn as_ptr(&self) -> *const T {
    self.inner.as_ref().unwrap().dptr as *const T
  }

  pub fn send(&self, other: &mut SyncDeviceBufferRefMut<'ctx, T>) {
    assert_eq!(self.len(), other.len());
    let self_dev_idx = self.inner.as_ref().unwrap().dev_idx;
    let other_dev_idx = other.inner.as_ref().unwrap().dev_idx;
    if self_dev_idx == other_dev_idx {
      // TODO(20151212)
      unimplemented!();
      /*unsafe { cuda_memcpy_async(
          other.as_mut_ptr() as *mut u8,
          self.as_ptr() as *const u8,
          self.size,
          CudaMemcpyKind::DeviceToDevice,
          &self.ctx.stream,
      ) }.unwrap();*/
    } else {
      unsafe { cuda_memcpy_peer_async(
          other.as_mut_ptr() as *mut u8, other_dev_idx,
          self.as_ptr() as *const u8, self_dev_idx,
          self.inner.as_ref().unwrap().size,
          &self.ctx.stream,
      ) };
    }
  }
}

pub struct SyncDeviceBufferRefMut<'ctx, T> where T: 'ctx + Copy {
  pub ctx:  &'ctx DeviceCtxRef<'ctx>,
  dev_sync: Arc<CudaEvent>,
  inner:    Option<MutexGuard<'ctx, InnerSyncDeviceBuffer<T>>>,
}

impl<'ctx, T> Drop for SyncDeviceBufferRefMut<'ctx, T> where T: 'ctx + Copy {
  fn drop(&mut self) {
    self.dev_sync.record(&self.ctx.stream).unwrap();
    let mut inner = self.inner.take().unwrap();
    assert!(inner.dev_sync.is_none());
    inner.dev_sync = Some(self.dev_sync.clone());
  }
}

impl<'ctx, T> SyncDeviceBufferRefMut<'ctx, T> where T: 'ctx + Copy {
  pub fn join(self) {
  }

  pub fn len(&self) -> usize {
    self.inner.as_ref().unwrap().len
  }

  pub unsafe fn as_ptr(&self) -> *const T {
    self.inner.as_ref().unwrap().dptr as *const T
  }

  pub unsafe fn as_mut_ptr(&self) -> *mut T {
    self.inner.as_ref().unwrap().dptr
  }

  pub fn recv_same(&mut self, other: &DeviceBufferRef<'ctx, T>) {
    assert_eq!(self.len(), other.len);
    let self_dev_idx = self.inner.as_ref().unwrap().dev_idx;
    if self_dev_idx == other.dev_idx {
      unsafe { cuda_memcpy_async(
          self.as_mut_ptr() as *mut u8,
          other.as_ptr() as *const u8,
          self.inner.as_ref().unwrap().size,
          CudaMemcpyKind::DeviceToDevice,
          &self.ctx.stream,
      ) }.unwrap();
    } else {
      // TODO(20151211)
      unimplemented!();
    }
  }
}*/

pub fn test_device_memory(buf1: &mut DeviceBuffer<f32>, buf2: &mut DeviceBuffer<f32>, ctx: &DeviceCtxRef) {
  let buf1 = buf1.borrow(ctx);
  let mut buf2 = buf2.borrow_mut(ctx);
  //buf1.send(&mut buf2);
  buf2.recv(&buf1);
}
