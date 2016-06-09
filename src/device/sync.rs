use device::context::{DeviceCtxRef};

use cuda::ffi::runtime::{cudaStream_t, cudaError_t};
use cuda::runtime::{CudaEvent};

use libc::{c_void};
use std::collections::{VecDeque};
use std::mem::{transmute};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

/*extern "C" fn shared_start_cb(stream: cudaStream_t, status: cudaError_t, user_data: *mut c_void) {
  match status {
    cudaError_t::Success => {}
    _ => panic!(),
  }
  let handle: Box<SharedDeviceCondvarHandle> = unsafe { transmute(user_data) };
  let mut q_ops = handle.q_ops.lock().unwrap();
  /*match (*q_ops.front(), handle.req_op) {
    (None, Op::ReadStart) |
    (Some(Op::ReadStart), Op::ReadStart) |
    (Some(Op::ReadComplete), Op::ReadStart) => {
    }
    (Some(Op::WriteStart), Op::ReadStart) |
    (Some(Op::WriteComplete), Op::ReadStart) => {
    }
    (None, Op::WriteStart) |
    (Some(Op::ReadStart), Op::WriteStart) |
    (Some(Op::ReadComplete), Op::WriteStart) => {
    }
    (Some(Op::WriteStart), Op::WriteStart) |
    (Some(Op::WriteComplete), Op::WriteStart) => {
    }
    _ => unreachable!(),
  }*/
  q_ops.push_back(handle.req_op);

  unimplemented!();
}

extern "C" fn shared_complete_cb(stream: cudaStream_t, status: cudaError_t, user_data: *mut c_void) {
  let handle: Box<SharedDeviceCondvarHandle> = unsafe { transmute(user_data) };
  let mut q_ops = handle.q_ops.lock().unwrap();
  q_ops.push_back(handle.req_op);

  unimplemented!();
}*/

extern "C" fn shared_post_cb(stream: cudaStream_t, status: cudaError_t, user_data: *mut c_void) {
  let handle: Box<SharedDeviceCondvarHandle> = unsafe { transmute(user_data) };

  unimplemented!();
}

extern "C" fn shared_wait_cb(stream: cudaStream_t, status: cudaError_t, user_data: *mut c_void) {
  let handle: Box<SharedDeviceCondvarHandle> = unsafe { transmute(user_data) };

  unimplemented!();
}

#[derive(Clone, Copy)]
enum Op {
  ReadStart,
  ReadComplete,
  WriteStart,
  WriteComplete,
}

pub struct SharedDeviceCondvarHandle {
  req_op:   Op,
  q_ops:    Arc<Mutex<VecDeque<Op>>>,
}

pub struct SharedDeviceCondvarState {
  sema_count:   AtomicUsize,
}

pub struct SharedDeviceCondvar {
  //dev_events:   Arc<Mutex<Vec<Option<CudaEvent>>>>,
}

impl SharedDeviceCondvar {
  pub fn start(&self) {
  }

  pub fn complete(&self) {
  }
}
