use context::{DeviceCtxRef};

use cuda::ffi::runtime::{
  cudaError,
  cudaFreeHost,
  cudaHostAlloc,
  cudaHostGetDevicePointer,
};
use cuda::runtime::{CudaEvent};

use libc::{c_void};
use std::cell::{RefCell};
use std::mem::{size_of};
use std::ptr::{Unique, null_mut};
use std::rc::{Rc};
use std::slice::{from_raw_parts, from_raw_parts_mut};

const WARP_SIZE: usize = 128;

#[derive(Clone, Copy)]
enum BufferState {
  Reset,
  Borrowed,
  Released,
}

/*pub struct MemoryBuffer<T> where T: Copy {
  dev_sync: Rc<RefCell<(BufferState, CudaEvent)>>,
  data: Vec<T>,
}*/

pub struct PageLockedBuffer<T> where T: Copy {
  dev_sync: Rc<RefCell<(BufferState, CudaEvent)>>,
  ptr:  Unique<T>,
  len:  usize,
  size: usize,
}

impl<T> Drop for PageLockedBuffer<T> where T: Copy {
  fn drop(&mut self) {
    self.sync();
    match unsafe { cudaFreeHost(*self.ptr as *mut c_void) } {
      cudaError::Success => {}
      e => {
        panic!("failed to deallocate pinned host memory: {:?}", e);
      }
    }
  }
}

impl<T> PageLockedBuffer<T> where T: Copy {
  pub fn new(len: usize, ctx: &DeviceCtxRef) -> PageLockedBuffer<T> {
    let min_size = len * size_of::<T>();
    let size = (min_size + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    let mut ptr: *mut c_void = null_mut();
    match unsafe { cudaHostAlloc(&mut ptr as *mut *mut c_void, size, 0) } {
      cudaError::Success => {}
      e => {
        panic!("failed to allocate PageLockedBuffer: {:?}", e);
      }
    }
    PageLockedBuffer{
      dev_sync: Rc::new(RefCell::new(
          (BufferState::Reset, CudaEvent::create_with_flags(0x02).unwrap())
      )),
      ptr:  unsafe { Unique::new(ptr as *mut T) },
      len:  len,
      size: size,
    }
  }

  pub fn len(&self) -> usize {
    self.len
  }

  fn sync(&self) {
    let mut dev_sync = self.dev_sync.borrow_mut();
    match dev_sync.0 {
      BufferState::Reset => {}
      BufferState::Released => {
        dev_sync.1.synchronize().unwrap();
        dev_sync.0 = BufferState::Reset;
      }
      _ => {
        panic!("invalid buffer state");
      }
    }
  }

  fn sync_borrow(&self) {
    let mut dev_sync = self.dev_sync.borrow_mut();
    match dev_sync.0 {
      BufferState::Reset => {
        dev_sync.0 = BufferState::Borrowed;
      }
      BufferState::Released => {
        dev_sync.1.synchronize().unwrap();
        dev_sync.0 = BufferState::Borrowed;
      }
      _ => {
        panic!("invalid buffer state");
      }
    }
  }

  pub fn as_slice(&self) -> &[T] {
    self.sync();
    unsafe { from_raw_parts(*self.ptr, self.len) }
  }

  pub fn as_mut_slice(&mut self) -> &mut [T] {
    self.sync();
    unsafe { from_raw_parts_mut(*self.ptr, self.len) }
  }

  pub fn borrow<'ctx>(&mut self, ctx: &'ctx DeviceCtxRef<'ctx>) -> HostBufferRef<'ctx, T> {
    self.sync_borrow();
    HostBufferRef{
      ctx:  ctx,
      dev_sync: self.dev_sync.clone(),
      ptr:  *self.ptr as *const T,
      len:  self.len,
      size: self.size,
    }
  }

  pub fn borrow_mut<'ctx>(&mut self, ctx: &'ctx DeviceCtxRef<'ctx>) -> HostBufferRefMut<'ctx, T> {
    self.sync_borrow();
    HostBufferRefMut{
      ctx:  ctx,
      dev_sync: self.dev_sync.clone(),
      ptr:  *self.ptr,
      len:  self.len,
      size: self.size,
    }
  }
}

pub struct PageLockedMappedBuffer<T> where T: Copy {
  dptr: *mut T,
  ptr:  Unique<T>,
  len:  usize,
  size: usize,
}

impl<T> Drop for PageLockedMappedBuffer<T> where T: Copy {
  fn drop(&mut self) {
    match unsafe { cudaFreeHost(*self.ptr as *mut c_void) } {
      cudaError::Success => {}
      e => {
        panic!("failed to deallocate pinned mapped host memory: {:?}", e);
      }
    }
  }
}

impl<T> PageLockedMappedBuffer<T> where T: Copy {
  pub fn new(len: usize, write_combined: bool, ctx: &DeviceCtxRef) -> PageLockedMappedBuffer<T> {
    let min_size = len * size_of::<T>();
    let size = (min_size + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    let mut ptr: *mut c_void = null_mut();
    let flags = 0x02 | if write_combined { 0x04 } else { 0 };
    match unsafe { cudaHostAlloc(&mut ptr as *mut *mut c_void, size, flags) } {
      cudaError::Success => {}
      e => {
        panic!("failed to allocated PageLockedMappedBuffer: {:?}", e);
      }
    }
    let mut dptr: *mut c_void = null_mut();
    match unsafe { cudaHostGetDevicePointer(&mut dptr as *mut *mut c_void, ptr as *mut c_void, 0) } {
      cudaError::Success => {}
      e => {
        panic!("failed to get device pointer: {:?}", e);
      }
    }
    PageLockedMappedBuffer{
      dptr: dptr as *mut T,
      ptr:  unsafe { Unique::new(ptr as *mut T) },
      len:  len,
      size: size,
    }
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn as_slice(&self) -> &[T] {
    unsafe { from_raw_parts(*self.ptr, self.len) }
  }

  pub fn as_mut_slice(&mut self) -> &mut [T] {
    unsafe { from_raw_parts_mut(*self.ptr, self.len) }
  }

  pub unsafe fn as_dev_ptr(&self) -> *const T {
    self.dptr as *const T
  }

  pub unsafe fn as_dev_mut_ptr(&mut self) -> *mut T {
    self.dptr
  }
}

pub struct HostBufferRef<'ctx, T> where T: 'ctx + Copy {
  ctx:  &'ctx DeviceCtxRef<'ctx>,
  dev_sync: Rc<RefCell<(BufferState, CudaEvent)>>,
  ptr:  *const T,
  len:  usize,
  size: usize,
}

impl<'ctx, T> Drop for HostBufferRef<'ctx, T> where T: 'ctx + Copy {
  fn drop(&mut self) {
    let mut dev_sync = self.dev_sync.borrow_mut();
    match dev_sync.0 {
      BufferState::Borrowed => {
        dev_sync.1.record(&self.ctx.stream).unwrap();
        dev_sync.0 = BufferState::Released;
      }
      _ => {
        panic!("invalid buffer state");
      }
    }
  }
}

impl<'ctx, T> HostBufferRef<'ctx, T> where T: 'ctx + Copy {
  pub fn len(&self) -> usize {
    self.len
  }

  pub unsafe fn as_ptr(&self) -> *const T {
    self.ptr
  }
}

pub struct HostBufferRefMut<'ctx, T> where T: 'ctx + Copy {
  ctx:  &'ctx DeviceCtxRef<'ctx>,
  dev_sync: Rc<RefCell<(BufferState, CudaEvent)>>,
  ptr:  *mut T,
  len:  usize,
  size: usize,
}

impl<'ctx, T> Drop for HostBufferRefMut<'ctx, T> where T: 'ctx + Copy {
  fn drop(&mut self) {
    let mut dev_sync = self.dev_sync.borrow_mut();
    match dev_sync.0 {
      BufferState::Borrowed => {
        dev_sync.1.record(&self.ctx.stream).unwrap();
        dev_sync.0 = BufferState::Released;
      }
      _ => {
        panic!("invalid buffer state");
      }
    }
  }
}

impl<'ctx, T> HostBufferRefMut<'ctx, T> where T: 'ctx + Copy {
  pub fn len(&self) -> usize {
    self.len
  }

  pub unsafe fn as_ptr(&self) -> *const T {
    self.ptr as *const T
  }

  pub unsafe fn as_mut_ptr(&self) -> *mut T {
    self.ptr
  }
}
