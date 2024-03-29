use device::array::{
  DeviceArray2dView, DeviceArray2dViewMut,
  DeviceArray3dView, DeviceArray3dViewMut,
};
use device::context::{DeviceCtxRef};
use device::num::{BytesExt, NumExt};
use host_memory::{HostBufferRef};

use array::{Shape};
use cuda::ffi::runtime::{
  cudaError_t,
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
use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

const WARP_SIZE: usize = 128;

/*pub trait DeviceStorage<T> where T: Copy {
  type Ref:     DeviceStorageRef<T>;
  type RefMut:  DeviceStorageRefMut<T>;

  fn as_ref<'ctx>(&mut self, ctx: &'ctx DeviceCtxRef<'ctx>) -> Self::Ref;
  fn as_ref_mut<'ctx>(&mut self, ctx: &'ctx DeviceCtxRef<'ctx>) -> Self::RefMut;
}*/

pub trait DeviceStorageRef<T> where T: Copy {
  fn device_idx(&self) -> usize;
  fn len(&self) -> usize;
  unsafe fn as_ptr(&self) -> *const T;
}

pub trait DeviceStorageRefMut<T>: DeviceStorageRef<T> where T: Copy {
  unsafe fn as_mut_ptr(&mut self) -> *mut T;
}

pub trait DeviceBufferInitExt {
  fn zeros(len: usize, ctx: &DeviceCtxRef) -> Self;
  fn ones(len: usize, ctx: &DeviceCtxRef) -> Self;
}

pub struct DeviceBuffer<T> where T: Copy {
  dev_sync: Rc<CudaEvent>,
  dev_idx:  usize,
  dptr: *mut T,
  len:  usize,
}

impl<T> Drop for DeviceBuffer<T> where T: Copy {
  fn drop(&mut self) {
    match unsafe { cudaFree(self.dptr as *mut c_void) } {
      cudaError_t::Success => {}
      cudaError_t::CudartUnloading => {
        // XXX(20160308): Sometimes drop() is called while the global runtime
        // is shutting down; suppress these errors.
      }
      e => {
        panic!("failed to free device memory: {:?}", e);
      }
    }
  }
}

impl<T> DeviceBuffer<T> where T: Copy {
  pub unsafe fn new(len: usize, ctx: &DeviceCtxRef) -> DeviceBuffer<T> {
    let min_size = len * size_of::<T>();
    let alloc_size = (min_size + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    let mut dptr: *mut c_void = null_mut();
    match unsafe { cudaMalloc(&mut dptr as *mut *mut c_void, alloc_size) } {
      cudaError_t::Success => {}
      e => {
        panic!("failed to allocate DeviceBuffer: {:?}", e);
      }
    }
    DeviceBuffer{
      dev_sync: Rc::new(CudaEvent::create_with_flags(0x02).unwrap()),
      dev_idx:  ctx.device(),
      dptr: dptr as *mut T,
      len:  len,
    }
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn view<'a, 'ctx>(&'a mut self, ctx: DeviceCtxRef<'ctx>) -> DeviceBufferView<'a, 'ctx, T> {
    ctx.stream.wait_event(&self.dev_sync).unwrap();
    DeviceBufferView{
      _marker:  PhantomData,
      ctx:      Some(ctx),
      dev_sync: self.dev_sync.clone(),
      dev_idx:  self.dev_idx,
      dptr:     self.dptr as *const T,
      len:      self.len,
    }
  }

  pub fn as_ref<'ctx>(&mut self, ctx: &'ctx DeviceCtxRef) -> DeviceBufferRef<'ctx, T> {
    ctx.stream.wait_event(&self.dev_sync).unwrap();
    DeviceBufferRef{
      ctx:  ctx,
      dev_sync: self.dev_sync.clone(),
      dev_idx:  self.dev_idx,
      dptr: self.dptr as *const T,
      len:  self.len,
    }
  }

  pub fn as_ref_range<'ctx>(&mut self, from: usize, to: usize, ctx: &'ctx DeviceCtxRef) -> DeviceBufferRef<'ctx, T> {
    assert!(from <= self.len);
    assert!(to <= self.len);
    assert!(from <= to);
    ctx.stream.wait_event(&self.dev_sync).unwrap();
    DeviceBufferRef{
      ctx:  ctx,
      dev_sync: self.dev_sync.clone(),
      dev_idx:  self.dev_idx,
      dptr: unsafe { (self.dptr as *const T).offset(from as isize) },
      len:  to - from,
    }
  }

  pub fn as_ref_mut<'ctx>(&mut self, ctx: &'ctx DeviceCtxRef) -> DeviceBufferRefMut<'ctx, T> {
    ctx.stream.wait_event(&self.dev_sync).unwrap();
    DeviceBufferRefMut{
      ctx:  ctx,
      dev_sync: self.dev_sync.clone(),
      dev_idx:  self.dev_idx,
      dptr: self.dptr,
      len:  self.len,
    }
  }

  pub fn as_ref_mut_range<'ctx>(&mut self, from: usize, to: usize, ctx: &'ctx DeviceCtxRef) -> DeviceBufferRefMut<'ctx, T> {
    assert!(from <= self.len);
    assert!(to <= self.len);
    assert!(from <= to);
    ctx.stream.wait_event(&self.dev_sync).unwrap();
    DeviceBufferRefMut{
      ctx:  ctx,
      dev_sync: self.dev_sync.clone(),
      dev_idx:  self.dev_idx,
      dptr: unsafe { self.dptr.offset(from as isize) },
      len:  to - from,
    }
  }
}

impl DeviceBufferInitExt for DeviceBuffer<u8> {
  fn zeros(len: usize, ctx: &DeviceCtxRef) -> DeviceBuffer<u8> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.as_ref_mut(ctx);
      buf_ref.set_memory(0);
    }
    buf
  }

  fn ones(len: usize, ctx: &DeviceCtxRef) -> DeviceBuffer<u8> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.as_ref_mut(ctx);
      buf_ref.set_memory(1);
    }
    buf
  }
}

impl DeviceBufferInitExt for DeviceBuffer<i32> {
  fn zeros(len: usize, ctx: &DeviceCtxRef) -> DeviceBuffer<i32> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.as_ref_mut(ctx);
      buf_ref.set_constant(0);
    }
    buf
  }

  fn ones(len: usize, ctx: &DeviceCtxRef) -> DeviceBuffer<i32> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.as_ref_mut(ctx);
      buf_ref.set_constant(0);
    }
    buf
  }
}

impl DeviceBufferInitExt for DeviceBuffer<f32> {
  fn zeros(len: usize, ctx: &DeviceCtxRef) -> DeviceBuffer<f32> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.as_ref_mut(ctx);
      buf_ref.set_constant(0.0);
    }
    buf
  }

  fn ones(len: usize, ctx: &DeviceCtxRef) -> DeviceBuffer<f32> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.as_ref_mut(ctx);
      buf_ref.set_constant(1.0);
    }
    buf
  }
}

pub struct DeviceBufferView<'a, 'ctx, T> where T: Copy {
  _marker:  PhantomData<&'a ()>,
  ctx:      Option<DeviceCtxRef<'ctx>>,
  dev_sync: Rc<CudaEvent>,
  dev_idx:  usize,
  dptr:     *const T,
  len:      usize,
}

impl<'a, 'ctx, T> Drop for DeviceBufferView<'a, 'ctx, T> where T: Copy {
  fn drop(&mut self) {
    if let Some(ref ctx) = self.ctx {
      self.dev_sync.record(&ctx.stream).unwrap();
    }
  }
}

impl<'a, 'ctx, T> DeviceBufferView<'a, 'ctx, T> where T: Copy {
  pub fn len(&self) -> usize {
    self.len
  }

  pub unsafe fn as_ptr(&self) -> *const T {
    self.dptr as *const T
  }

  pub fn range(mut self, from: usize, to: usize) -> DeviceBufferView<'a, 'ctx, T> {
    assert!(from <= self.len);
    assert!(to <= self.len);
    assert!(from <= to);
    self.ctx.as_ref().unwrap().stream.wait_event(&self.dev_sync).unwrap();
    DeviceBufferView{
      _marker:  PhantomData,
      ctx:      self.ctx.take(),
      dev_sync: self.dev_sync.clone(),
      dev_idx:  self.dev_idx,
      dptr:     unsafe { (self.dptr as *const T).offset(from as isize) },
      len:      to - from,
    }
  }
}

pub struct DeviceBufferRef<'ctx, T> where T: 'ctx + Copy {
  pub ctx:  &'ctx DeviceCtxRef<'ctx>,
  dev_sync: Rc<CudaEvent>,
  dev_idx:  usize,
  dptr: *const T,
  len:  usize,
}

impl<'ctx, T> Drop for DeviceBufferRef<'ctx, T> where T: 'ctx + Copy {
  fn drop(&mut self) {
    self.dev_sync.record(&self.ctx.stream).unwrap();
  }
}

/*impl<'ctx, T> DeviceStorageRef<T> for DeviceBufferRef<'ctx, T> where T: 'ctx + Copy {
  pub unsafe fn as_ptr(&self) -> *const T {
    self.dptr as *const T
  }
}*/

impl<'ctx, T> DeviceBufferRef<'ctx, T> where T: 'ctx + Copy {
  pub fn len(&self) -> usize {
    self.len
  }

  pub unsafe fn as_ptr(&self) -> *const T {
    self.dptr as *const T
  }

  pub fn range(&self, from: usize, to: usize) -> DeviceBufferRef<'ctx, T> {
    assert!(from <= self.len);
    assert!(to <= self.len);
    assert!(from <= to);
    self.ctx.stream.wait_event(&self.dev_sync).unwrap();
    DeviceBufferRef{
      ctx:  self.ctx,
      dev_sync: self.dev_sync.clone(),
      dev_idx:  self.dev_idx,
      dptr: unsafe { (self.dptr as *const T).offset(from as isize) },
      len:  to - from,
    }
  }

  pub fn into_2d_view(self, bound: (usize, usize)) -> DeviceArray2dView<'ctx, T> {
    // FIXME(20160201): should take a range first for exact size.
    assert!(bound.len() <= self.len);
    DeviceArray2dView{
      data:     self,
      bound:    bound,
      stride:   bound.to_least_stride(),
    }
  }

  pub fn into_3d_view(self, bound: (usize, usize, usize)) -> DeviceArray3dView<'ctx, T> {
    // FIXME(20160201): should take a range first for exact size.
    assert!(bound.len() <= self.len);
    DeviceArray3dView{
      data:     self,
      bound:    bound,
      stride:   bound.to_least_stride(),
    }
  }

  pub fn send(&self, other: &mut DeviceBufferRefMut<'ctx, T>) {
    assert_eq!(self.len, other.len);
    //assert_eq!(self.size, other.size);
    if self.dev_idx == other.dev_idx {
      unsafe { cuda_memcpy_async(
          other.as_mut_ptr() as *mut u8,
          self.as_ptr() as *const u8,
          self.len * size_of::<T>(),
          CudaMemcpyKind::DeviceToDevice,
          &self.ctx.stream,
      ) }.unwrap();
    } else {
      // TODO(20151211)
      unimplemented!();
    }
  }

  pub fn raw_send<'a>(&self, other: &RawDeviceBufferRef<'a, T>) {
    assert_eq!(self.len, other.len);
    //assert_eq!(self.size, other.size);
    if self.dev_idx == other.dev_idx {
      unsafe { cuda_memcpy_async(
          other.as_mut_ptr() as *mut u8,
          self.as_ptr() as *const u8,
          self.len * size_of::<T>(),
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

  pub unsafe fn unsafe_async_store(&mut self, host_buf: &mut [T]) {
    assert_eq!(self.len, host_buf.len());
    unsafe { cuda_memcpy_async(
        host_buf.as_mut_ptr() as *mut u8,
        self.as_ptr() as *const u8,
        self.len * size_of::<T>(),
        CudaMemcpyKind::DeviceToHost,
        &self.ctx.stream,
    ) }.unwrap();
  }
}

/*pub struct DeviceFuture<'a, 'ctx> {
  _marker:  PhantomData<&'a ()>,
  ctx:      &'ctx DeviceCtxRef<'ctx>,
}

impl<'a, 'ctx> Drop for DeviceFuture<'a, 'ctx> {
  fn drop(&mut self) {
    self.ctx.blocking_sync();
  }
}*/

pub struct DeviceBufferRefMut<'ctx, T> where T: 'ctx + Copy {
  pub ctx:  &'ctx DeviceCtxRef<'ctx>,
  dev_sync: Rc<CudaEvent>,
  dev_idx:  usize,
  dptr: *mut T,
  len:  usize,
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

  pub fn mut_range(&mut self, from: usize, to: usize) -> DeviceBufferRefMut<'ctx, T> {
    assert!(from <= self.len);
    assert!(to <= self.len);
    assert!(from <= to);
    self.ctx.stream.wait_event(&self.dev_sync).unwrap();
    DeviceBufferRefMut{
      ctx:  self.ctx,
      dev_sync: self.dev_sync.clone(),
      dev_idx:  self.dev_idx,
      dptr: unsafe { self.dptr.offset(from as isize) },
      len:  to - from,
    }
  }

  pub fn into_2d_view_mut(self, bound: (usize, usize)) -> DeviceArray2dViewMut<'ctx, T> {
    // FIXME(20160201): should take a range first for exact size.
    assert!(bound.len() <= self.len);
    DeviceArray2dViewMut{
      data:     self,
      bound:    bound,
      stride:   bound.to_least_stride(),
    }
  }

  pub fn into_3d_view_mut(self, bound: (usize, usize, usize)) -> DeviceArray3dViewMut<'ctx, T> {
    // FIXME(20160201): should take a range first for exact size.
    assert!(bound.len() <= self.len);
    DeviceArray3dViewMut{
      data:     self,
      bound:    bound,
      stride:   bound.to_least_stride(),
    }
  }

  pub fn copy(&mut self, src: &DeviceBufferRef<'ctx, T>) {
    assert_eq!(self.len, src.len);
    if self.dev_idx == src.dev_idx {
      unsafe { cuda_memcpy_async(
          self.as_mut_ptr() as *mut u8,
          src.as_ptr() as *const u8,
          self.len * size_of::<T>(),
          CudaMemcpyKind::DeviceToDevice,
          &self.ctx.stream,
      ) }.unwrap();
    } else {
      // TODO(20151211)
      unimplemented!();
    }
  }

  pub fn copy_raw(&mut self, src: &RawDeviceBufferRef<'ctx, T>) {
    assert_eq!(self.len, src.len);
    if self.dev_idx == src.dev_idx {
      unsafe { cuda_memcpy_async(
          self.as_mut_ptr() as *mut u8,
          src.as_ptr() as *const u8,
          self.len * size_of::<T>(),
          CudaMemcpyKind::DeviceToDevice,
          &self.ctx.stream,
      ) }.unwrap();
    } else {
      // TODO(20151211)
      unimplemented!();
    }
  }

  pub fn recv(&mut self, other: &DeviceBufferRef<'ctx, T>) {
    assert_eq!(self.len, other.len);
    if self.dev_idx == other.dev_idx {
      unsafe { cuda_memcpy_async(
          self.as_mut_ptr() as *mut u8,
          other.as_ptr() as *const u8,
          self.len * size_of::<T>(),
          CudaMemcpyKind::DeviceToDevice,
          &self.ctx.stream,
      ) }.unwrap();
    } else {
      // TODO(20151211)
      unimplemented!();
    }
  }

  pub fn raw_recv<'a>(&mut self, src: &RawDeviceBufferRef<'a, T>) {
    assert_eq!(self.len, src.len);
    //assert_eq!(self.size, src.size);
    if self.dev_idx == src.dev_idx {
      unsafe { cuda_memcpy_async(
          self.as_mut_ptr() as *mut u8,
          src.as_ptr() as *const u8,
          self.len * size_of::<T>(),
          CudaMemcpyKind::DeviceToDevice,
          &self.ctx.stream,
      ) }.unwrap();
    } else {
      unsafe { cuda_memcpy_peer_async(
          self.as_mut_ptr() as *mut u8, self.dev_idx,
          src.as_ptr() as *const u8, src.dev_idx,
          //self.size,
          self.len * size_of::<T>(),
          &self.ctx.stream,
      ) };
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

  /*pub fn async_load<'a>(&mut self, host_buf: &'a [T]) -> DeviceFuture<'a, 'ctx> {
    assert_eq!(self.len, host_buf.len());
    unsafe { cuda_memcpy_async(
        self.as_mut_ptr() as *mut u8,
        host_buf.as_ptr() as *const u8,
        self.len * size_of::<T>(),
        CudaMemcpyKind::HostToDevice,
        &self.ctx.stream,
    ) }.unwrap();
    DeviceFuture{
      _marker:  PhantomData,
      ctx:      self.ctx,
    }
  }*/

  pub unsafe fn unsafe_async_load(&mut self, host_buf: &[T]) {
    assert_eq!(self.len, host_buf.len());
    unsafe { cuda_memcpy_async(
        self.as_mut_ptr() as *mut u8,
        host_buf.as_ptr() as *const u8,
        self.len * size_of::<T>(),
        CudaMemcpyKind::HostToDevice,
        &self.ctx.stream,
    ) }.unwrap();
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

pub struct SharedDeviceBuffer<T> where T: Copy {
  lock:     Arc<RwLock<()>>,
  dev_sync: Arc<Mutex<CudaEvent>>,
  //dev_sync: Arc<Mutex<CudaEvent>>,
  //dev_sync: Arc<CudaEvent>,
  dev_idx:  usize,
  dptr:     *mut T,
  len:      usize,
}

unsafe impl<T> Send for SharedDeviceBuffer<T> where T: Copy {}
unsafe impl<T> Sync for SharedDeviceBuffer<T> where T: Copy {}

impl<T> Drop for SharedDeviceBuffer<T> where T: Copy {
  fn drop(&mut self) {
    match unsafe { cudaFree(self.dptr as *mut c_void) } {
      cudaError_t::Success => {}
      cudaError_t::CudartUnloading => {
        // XXX(20160308): Sometimes drop() is called while the global runtime
        // is shutting down; suppress these errors.
      }
      e => {
        panic!("failed to free device memory: {:?}", e);
      }
    }
  }
}

impl<T> SharedDeviceBuffer<T> where T: Copy {
  pub unsafe fn new(len: usize, ctx: &DeviceCtxRef) -> SharedDeviceBuffer<T> {
    let min_size = len * size_of::<T>();
    let alloc_size = (min_size + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    let mut dptr: *mut c_void = null_mut();
    match unsafe { cudaMalloc(&mut dptr as *mut *mut c_void, alloc_size) } {
      cudaError_t::Success => {}
      e => {
        panic!("failed to allocate DeviceBuffer: {:?}", e);
      }
    }
    SharedDeviceBuffer{
      lock:     Arc::new(RwLock::new(())),
      //dev_sync: Arc::new(RwLock::new(CudaEvent::create_with_flags(0x02).unwrap())),
      dev_sync: Arc::new(Mutex::new(CudaEvent::create_with_flags(0x02).unwrap())),
      //dev_sync: Arc::new(CudaEvent::create_with_flags(0x02).unwrap()),
      dev_idx:  ctx.device(),
      dptr:     dptr as *mut T,
      len:      len,
    }
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn as_ref<'ctx>(&'ctx self, ctx: &'ctx DeviceCtxRef) -> SharedDeviceBufferRef<'ctx, T> {
    ctx.stream.wait_event(&*self.dev_sync.lock().unwrap()).unwrap();
    SharedDeviceBufferRef{
      ctx:      ctx,
      guard:    self.lock.read().unwrap(),
      dev_sync: self.dev_sync.clone(),
      dev_idx:  self.dev_idx,
      dptr:     self.dptr as *const T,
      len:      self.len,
    }
  }

  /*pub fn as_ref_range<'ctx>(&mut self, from: usize, to: usize, ctx: &'ctx DeviceCtxRef) -> DeviceBufferRef<'ctx, T> {
    assert!(from <= self.len);
    assert!(to <= self.len);
    assert!(from <= to);
    ctx.stream.wait_event(&self.dev_sync).unwrap();
    DeviceBufferRef{
      ctx:  ctx,
      dev_sync: self.dev_sync.clone(),
      dev_idx:  self.dev_idx,
      dptr: unsafe { (self.dptr as *const T).offset(from as isize) },
      len:  to - from,
      //size: (to - from) * size_of::<T>(),
    }
  }*/

  pub fn as_ref_mut<'ctx>(&'ctx self, ctx: &'ctx DeviceCtxRef) -> SharedDeviceBufferRefMut<'ctx, T> {
    ctx.stream.wait_event(&*self.dev_sync.lock().unwrap()).unwrap();
    SharedDeviceBufferRefMut{
      ctx:      ctx,
      guard:    self.lock.write().unwrap(),
      dev_sync: self.dev_sync.clone(),
      dev_idx:  self.dev_idx,
      dptr:     self.dptr,
      len:      self.len,
    }
  }

  /*pub fn as_ref_mut_range<'ctx>(&mut self, from: usize, to: usize, ctx: &'ctx DeviceCtxRef) -> DeviceBufferRefMut<'ctx, T> {
    assert!(from <= self.len);
    assert!(to <= self.len);
    assert!(from <= to);
    ctx.stream.wait_event(&self.dev_sync).unwrap();
    DeviceBufferRefMut{
      ctx:  ctx,
      dev_sync: self.dev_sync.clone(),
      dev_idx:  self.dev_idx,
      dptr: unsafe { self.dptr.offset(from as isize) },
      len:  to - from,
      //size: (to - from) * size_of::<T>(),
    }
  }*/
}

/*impl DeviceBufferInitExt for SharedDeviceBuffer<u8> {
  fn zeros(len: usize, ctx: &DeviceCtxRef) -> SharedDeviceBuffer<u8> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.as_ref_mut(ctx);
      buf_ref.set_memory(0);
    }
    buf
  }

  fn ones(len: usize, ctx: &DeviceCtxRef) -> SharedDeviceBuffer<u8> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.as_ref_mut(ctx);
      buf_ref.set_memory(1);
    }
    buf
  }
}

impl DeviceBufferInitExt for SharedDeviceBuffer<i32> {
  fn zeros(len: usize, ctx: &DeviceCtxRef) -> SharedDeviceBuffer<i32> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.as_ref_mut(ctx);
      buf_ref.set_constant(0);
    }
    buf
  }

  fn ones(len: usize, ctx: &DeviceCtxRef) -> SharedDeviceBuffer<i32> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.as_ref_mut(ctx);
      buf_ref.set_constant(0);
    }
    buf
  }
}

impl DeviceBufferInitExt for SharedDeviceBuffer<f32> {
  fn zeros(len: usize, ctx: &DeviceCtxRef) -> SharedDeviceBuffer<f32> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.as_ref_mut(ctx);
      buf_ref.set_constant(0.0);
    }
    buf
  }

  fn ones(len: usize, ctx: &DeviceCtxRef) -> SharedDeviceBuffer<f32> {
    let mut buf = unsafe { Self::new(len, ctx) };
    {
      let mut buf_ref = buf.as_ref_mut(ctx);
      buf_ref.set_constant(1.0);
    }
    buf
  }
}*/

pub struct SharedDeviceBufferRef<'ctx, T> where T: 'ctx + Copy {
  pub ctx:  &'ctx DeviceCtxRef<'ctx>,
  guard:    RwLockReadGuard<'ctx, ()>,
  dev_sync: Arc<Mutex<CudaEvent>>,
  dev_idx:  usize,
  dptr:     *const T,
  len:      usize,
}

impl<'ctx, T> Drop for SharedDeviceBufferRef<'ctx, T> where T: 'ctx + Copy {
  fn drop(&mut self) {
    self.dev_sync.lock().unwrap().record(&self.ctx.stream).unwrap();
  }
}

impl<'ctx, T> SharedDeviceBufferRef<'ctx, T> where T: 'ctx + Copy {
  pub fn len(&self) -> usize {
    self.len
  }

  pub unsafe fn as_ptr(&self) -> *const T {
    self.dptr as *const T
  }
}

pub struct SharedDeviceBufferRefMut<'ctx, T> where T: 'ctx + Copy {
  pub ctx:  &'ctx DeviceCtxRef<'ctx>,
  guard:    RwLockWriteGuard<'ctx, ()>,
  dev_sync: Arc<Mutex<CudaEvent>>,
  dev_idx:  usize,
  dptr:     *mut T,
  len:      usize,
}

impl<'ctx, T> Drop for SharedDeviceBufferRefMut<'ctx, T> where T: 'ctx + Copy {
  fn drop(&mut self) {
    self.dev_sync.lock().unwrap().record(&self.ctx.stream).unwrap();
  }
}

impl<'ctx, T> SharedDeviceBufferRefMut<'ctx, T> where T: 'ctx + Copy {
  pub fn len(&self) -> usize {
    self.len
  }

  pub unsafe fn as_ptr(&self) -> *const T {
    self.dptr as *const T
  }

  pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
    self.dptr
  }

  pub fn copy_generic<U>(&mut self, src: &U) where U: DeviceStorageRef<T> {
    assert_eq!(self.len, src.len());
    if self.dev_idx == src.device_idx() {
      unsafe { cuda_memcpy_async(
          self.as_mut_ptr() as *mut u8,
          src.as_ptr() as *const u8,
          self.len * size_of::<T>(),
          CudaMemcpyKind::DeviceToDevice,
          &self.ctx.stream,
      ) }.unwrap();
    } else {
      unsafe { cuda_memcpy_peer_async(
          self.as_mut_ptr() as *mut u8, self.dev_idx,
          src.as_ptr() as *const u8, src.device_idx(),
          self.len * size_of::<T>(),
          &self.ctx.stream,
      ) }.unwrap();
    }
  }

  pub fn copy_single(&mut self, src: &DeviceBufferRef<T>) {
    assert_eq!(self.len, src.len());
    if self.dev_idx == src.dev_idx {
      unsafe { cuda_memcpy_async(
          self.as_mut_ptr() as *mut u8,
          src.as_ptr() as *const u8,
          self.len * size_of::<T>(),
          CudaMemcpyKind::DeviceToDevice,
          &self.ctx.stream,
      ) }.unwrap();
    } else {
      unsafe { cuda_memcpy_peer_async(
          self.as_mut_ptr() as *mut u8, self.dev_idx,
          src.as_ptr() as *const u8, src.dev_idx,
          self.len * size_of::<T>(),
          &self.ctx.stream,
      ) }.unwrap();
    }
  }

  pub fn copy(&mut self, src: &SharedDeviceBufferRef<T>) {
    assert_eq!(self.len, src.len());
    if self.dev_idx == src.dev_idx {
      unsafe { cuda_memcpy_async(
          self.as_mut_ptr() as *mut u8,
          src.as_ptr() as *const u8,
          self.len * size_of::<T>(),
          CudaMemcpyKind::DeviceToDevice,
          &self.ctx.stream,
      ) }.unwrap();
    } else {
      unsafe { cuda_memcpy_peer_async(
          self.as_mut_ptr() as *mut u8, self.dev_idx,
          src.as_ptr() as *const u8, src.dev_idx,
          self.len * size_of::<T>(),
          &self.ctx.stream,
      ) }.unwrap();
    }
  }
}

pub struct RawDeviceBuffer<T> where T: Copy {
  dev_idx:  usize,
  dptr:     *mut T,
  len:      usize,
  //size:     usize,
}

unsafe impl<T> Send for RawDeviceBuffer<T> where T: Copy {}
unsafe impl<T> Sync for RawDeviceBuffer<T> where T: Copy {}

impl<T> Drop for RawDeviceBuffer<T> where T: Copy {
  fn drop(&mut self) {
    match unsafe { cudaFree(self.dptr as *mut c_void) } {
      cudaError_t::Success => {}
      cudaError_t::CudartUnloading => {
        // XXX(20160308): Sometimes drop() is called while the global runtime
        // is shutting down; suppress these errors.
      }
      e => {
        panic!("failed to free device memory: {:?}", e);
      }
    }
  }
}

impl<T> RawDeviceBuffer<T> where T: Copy {
  pub unsafe fn new(len: usize, ctx: &DeviceCtxRef) -> RawDeviceBuffer<T> {
    let dev_idx = ctx.device();
    // FIXME(20160417): for debugging.
    //assert_eq!(dev_idx, 0);
    let min_size = len * size_of::<T>();
    let size = (min_size + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    let mut dptr: *mut c_void = null_mut();
    match unsafe { cudaMalloc(&mut dptr as *mut *mut c_void, size) } {
      cudaError_t::Success => {}
      e => {
        panic!("failed to allocate DeviceBuffer: {:?}", e);
      }
    }
    RawDeviceBuffer{
      dev_idx:  dev_idx,
      dptr:     dptr as *mut T,
      len:      len,
      //size:     size,
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

  pub fn raw_send<'ctx>(&self, other: &RawDeviceBuffer<T>, ctx: &DeviceCtxRef<'ctx>) {
    assert_eq!(self.len, other.len);
    //assert_eq!(self.size, other.size);
    if self.dev_idx == other.dev_idx {
      unsafe { cuda_memcpy_async(
          other.as_mut_ptr() as *mut u8,
          self.as_ptr() as *const u8,
          self.len * size_of::<T>(),
          CudaMemcpyKind::DeviceToDevice,
          &ctx.stream,
      ) }.unwrap();
    } else {
      unsafe { cuda_memcpy_peer_async(
          other.as_mut_ptr() as *mut u8, other.dev_idx,
          self.as_ptr() as *const u8, self.dev_idx,
          self.len * size_of::<T>(),
          &ctx.stream,
      ) }.unwrap();
    }
  }

  pub fn sync_store<'ctx>(&self, host_buf: &mut [T], ctx: &DeviceCtxRef<'ctx>) {
    assert_eq!(self.len, host_buf.len());
    unsafe { cuda_memcpy_async(
        host_buf.as_mut_ptr() as *mut u8,
        self.as_ptr() as *const u8,
        self.len * size_of::<T>(),
        CudaMemcpyKind::DeviceToHost,
        &ctx.stream,
    ) }.unwrap();
    ctx.blocking_sync();
  }

  pub fn sync_load<'ctx>(&mut self, host_buf: &[T], ctx: &DeviceCtxRef<'ctx>) {
    assert_eq!(self.len, host_buf.len());
    unsafe { cuda_memcpy_async(
        self.as_mut_ptr() as *mut u8,
        host_buf.as_ptr() as *const u8,
        self.len * size_of::<T>(),
        CudaMemcpyKind::HostToDevice,
        &ctx.stream,
    ) }.unwrap();
    ctx.blocking_sync();
  }

  pub fn as_ref<'a>(&'a self) -> RawDeviceBufferRef<'a, T> {
    RawDeviceBufferRef{
      dev_idx:  self.dev_idx,
      dptr:     self.dptr,
      len:      self.len,
      //size:     self.size,
      _marker:  PhantomData,
    }
  }

  pub fn as_ref_range<'a>(&'a self, from: usize, to: usize) -> RawDeviceBufferRef<'a, T> {
    assert!(from <= self.len);
    assert!(to <= self.len);
    assert!(from <= to);
    RawDeviceBufferRef{
      dev_idx:  self.dev_idx,
      dptr:     unsafe { self.dptr.offset(from as isize) },
      len:      to - from,
      //size:     (to - from) * size_of::<T>(),
      _marker:  PhantomData,
    }
  }
}

pub struct RawDeviceBufferRef<'a, T> where T: Copy {
  dev_idx:  usize,
  dptr:     *mut T,
  len:      usize,
  //size:     usize,
  _marker:  PhantomData<&'a ()>,
}

impl<'a, T> RawDeviceBufferRef<'a, T> where T: Copy {
  pub fn len(&self) -> usize {
    self.len
  }

  pub unsafe fn as_ptr(&self) -> *const T {
    self.dptr as *const T
  }

  pub unsafe fn as_mut_ptr(&self) -> *mut T {
    self.dptr
  }

  pub fn copy(&mut self, src: &DeviceBufferRef<T>) {
    assert_eq!(self.len, src.len);
    if self.dev_idx == src.dev_idx {
      unsafe { cuda_memcpy_async(
          self.as_mut_ptr() as *mut u8,
          src.as_ptr() as *const u8,
          self.len * size_of::<T>(),
          CudaMemcpyKind::DeviceToDevice,
          &src.ctx.stream,
      ) }.unwrap();
    } else {
      // TODO(20151211)
      unimplemented!();
    }
  }

  pub fn copy_raw(&mut self, src: &RawDeviceBufferRef<T>, ctx: &DeviceCtxRef) {
    assert_eq!(self.len, src.len);
    if self.dev_idx == src.dev_idx {
      unsafe { cuda_memcpy_async(
          self.as_mut_ptr() as *mut u8,
          src.as_ptr() as *const u8,
          self.len * size_of::<T>(),
          CudaMemcpyKind::DeviceToDevice,
          &ctx.stream,
      ) }.unwrap();
    } else {
      // TODO(20151211)
      unimplemented!();
    }
  }
}

pub fn test_device_memory(buf1: &mut DeviceBuffer<f32>, buf2: &mut DeviceBuffer<f32>, ctx: &DeviceCtxRef) {
  let buf1 = buf1.as_ref(ctx);
  let mut buf2 = buf2.as_ref_mut(ctx);
  //buf1.send(&mut buf2);
  buf2.recv(&buf1);
}
