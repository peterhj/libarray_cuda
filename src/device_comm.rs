use context::{DeviceCtxRef};
use device_ext::*;
use device_memory::{DeviceBufferRef, SyncDeviceBuffer, SyncDeviceBufferRef, SyncDeviceBufferRefMut};
use ffi::*;

use libc::{c_int};

pub trait ReduceOp {
  fn reduce<'ctx>(src: &mut SyncDeviceBufferRef<'ctx, Self>, dst: &mut SyncDeviceBufferRefMut<'ctx, Self>) where Self: Copy;
}

impl ReduceOp for i32 {
  fn reduce<'ctx>(src: &mut SyncDeviceBufferRef<'ctx, i32>, dst: &mut SyncDeviceBufferRefMut<'ctx, i32>) {
    unsafe { array_cuda_map_add_i32(
        src.as_ptr(), src.len() as c_int,
        dst.as_mut_ptr(),
        dst.ctx.stream.ptr,
    ) };
  }
}

pub struct DeviceReduceWorker<T> where T: Copy + ReduceOp {
  tid:  usize,
  nths: usize,
  width:    usize,
  depth:    usize,
  src_bufs: Vec<Option<SyncDeviceBuffer<T>>>,
  dst_bufs: Vec<Option<SyncDeviceBuffer<T>>>,
}

impl<T> DeviceReduceWorker<T> where T: Copy + ReduceOp {
  pub fn new(tid: usize, num_threads: usize, bufs: &[SyncDeviceBuffer<T>]) -> DeviceReduceWorker<T> {
    assert_eq!(2 * num_threads, bufs.len());
    let width = num_threads.checked_next_power_of_two().unwrap();
    let depth = width.trailing_zeros() as usize;
    //println!("DEBUG: tid: {} width: {} depth: {}", tid, width, depth);
    let mut src_bufs = vec![];
    let mut dst_bufs = vec![];
    // TODO(20151212): permute the buffers so that peer access enabled pairs are
    // spatially adjacent.
    for d in (0 .. depth) {
      let r = 1 << (d + 1);
      let half_r = r / 2;
      //println!("DEBUG: tid: {} r: {} half_r: {}", tid, r, half_r);
      //if tid < r {
        if (tid % r) == half_r {
          //println!("DEBUG: tid: {} push bufs: {}", tid, src_bufs.len());
          src_bufs.push(Some(bufs[tid].clone()));
          dst_bufs.push(Some(bufs[num_threads + tid - half_r].clone()));
        } else if (tid % r) == 0 {
          //println!("DEBUG: tid: {} push bufs: {}", tid, src_bufs.len());
          src_bufs.push(Some(bufs[tid].clone()));
          dst_bufs.push(Some(bufs[num_threads + tid].clone()));
        } else {
          src_bufs.push(None);
          dst_bufs.push(None);
        }
      /*} else {
        src_bufs.push(None);
        dst_bufs.push(None);
      }*/
    }
    assert_eq!(src_bufs.len(), dst_bufs.len());
    DeviceReduceWorker{
      tid:  tid,
      nths: num_threads,
      width:    width,
      depth:    depth,
      src_bufs: src_bufs,
      dst_bufs: dst_bufs,
    }
  }

  pub fn process(&mut self, input: &mut DeviceBufferRef<T>, ctx: &DeviceCtxRef) {
    let tid = self.tid;
    let depth = self.depth;
    let &mut DeviceReduceWorker{
      ref mut src_bufs, ref mut dst_bufs, .. } = self;
    let mut guards = vec![];

    if depth >= 1 {
      assert!(src_bufs[0].is_some());
      let src_buf = src_bufs[0].as_mut().unwrap();
      let (mut src_ref, src_guard) = src_buf.borrow_mut(ctx);
      src_ref.recv_same(input);
      guards.push(src_guard);
    }

    for d in (0 .. depth) {
      let r = 1 << (d + 1);
      let half_r = r / 2;
      //if tid < r {
        if (tid % r) == half_r {
          assert!(src_bufs[d].is_some());
          assert!(dst_bufs[d].is_some());
          let src_buf = src_bufs[d].as_mut().unwrap();
          let dst_buf = dst_bufs[d].as_mut().unwrap();
          let (src_ref, src_guard) = src_buf.borrow(ctx);
          let (mut dst_ref, dst_guard) = dst_buf.borrow_mut(ctx);
          src_ref.send(&mut dst_ref);
          guards.push(src_guard);
          guards.push(dst_guard);
        } else if (tid % r) == 0 {
          assert!(src_bufs[d].is_some());
          assert!(dst_bufs[d].is_some());
          let src_buf = src_bufs[d].as_mut().unwrap();
          let dst_buf = dst_bufs[d].as_mut().unwrap();
          let (mut dst_ref, dst_guard) = dst_buf.borrow(ctx);
          let (mut src_ref, src_guard) = src_buf.borrow_mut(ctx);
          ReduceOp::reduce(&mut dst_ref, &mut src_ref);
          guards.push(dst_guard);
          guards.push(src_guard);
        }
      //}
    }

    guards.clear();

    if tid == 0 {
      assert!(src_bufs[0].is_some());
      let src_buf = src_bufs[0].as_mut().unwrap();
      let (mut src_ref, _) = src_buf.borrow(ctx);
      unsafe { array_cuda_map_print_i32(
          src_ref.as_ptr() as *const i32, src_ref.len() as c_int,
          ctx.stream.ptr,
      ) };
    }
  }
}

pub struct DeviceAllReduceWorker<T> where T: Copy {
  tid:  usize,
  nths: usize,
  width:    usize,
  depth:    usize,
  src_bufs: Vec<Option<SyncDeviceBuffer<T>>>,
  dst_bufs: Vec<Option<SyncDeviceBuffer<T>>>,
}

impl<T> DeviceAllReduceWorker<T> where T: Copy {
  pub fn new(tid: usize, num_threads: usize, bufs: &[SyncDeviceBuffer<T>]) -> DeviceAllReduceWorker<T> {
    unimplemented!();
  }
}
