use context::{DeviceCtxRef};
use device_ext::*;
use device_memory::{
  DeviceBufferRef, RawDeviceBuffer,
  SyncDeviceBuffer, SyncDeviceBufferRef, SyncDeviceBufferRefMut,
};
use ffi::*;

use libc::{c_int};
use std::sync::{Arc, Barrier};

pub trait ReduceOp {
  fn reduce<'ctx>(src: &RawDeviceBuffer<Self>, dst: &RawDeviceBuffer<Self>, ctx: &DeviceCtxRef<'ctx>) where Self: Copy;
}

impl ReduceOp for i32 {
  fn reduce<'ctx>(src: &RawDeviceBuffer<i32>, dst: &RawDeviceBuffer<i32>, ctx: &DeviceCtxRef<'ctx>) {
    unsafe { array_cuda_map_add_i32(
        src.as_ptr(), src.len() as c_int,
        dst.as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }
}

impl ReduceOp for f32 {
  fn reduce<'ctx>(src: &RawDeviceBuffer<f32>, dst: &RawDeviceBuffer<f32>, ctx: &DeviceCtxRef<'ctx>) {
    unsafe { array_cuda_map_add_f32(
        src.as_ptr(), src.len() as c_int,
        dst.as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }
}

pub struct DeviceReduceWorker<T> where T: Copy + ReduceOp {
  tid:  usize,
  nths: usize,
  barrier:  Arc<Barrier>,
  width:    usize,
  depth:    usize,
  src_bufs: Vec<Option<Arc<RawDeviceBuffer<T>>>>,
  dst_bufs: Vec<Option<Arc<RawDeviceBuffer<T>>>>,
  //mrg_bufs: Vec<Option<Arc<RawDeviceBuffer<T>>>>,
}

impl<T> DeviceReduceWorker<T> where T: Copy + ReduceOp {
  pub fn new(tid: usize, num_threads: usize, barrier: Arc<Barrier>, bufs: &[Arc<RawDeviceBuffer<T>>]) -> DeviceReduceWorker<T> {
    assert_eq!(2 * num_threads, bufs.len());
    let width = num_threads.checked_next_power_of_two().unwrap();
    let depth = width.trailing_zeros() as usize;
    let mut src_bufs = vec![];
    let mut dst_bufs = vec![];
    //let mut mrg_bufs = vec![];
    // TODO(20151212): permute the buffers so that peer access enabled pairs are
    // spatially adjacent.
    for d in (0 .. depth) {
      let r = 1 << (d + 1);
      let half_r = r / 2;
      if (tid % r) == half_r {
        src_bufs.push(Some(bufs[tid].clone()));
        dst_bufs.push(Some(bufs[num_threads + tid - half_r].clone()));
        //mrg_bufs.push(None);
      } else if (tid % r) == 0 {
        src_bufs.push(Some(bufs[tid].clone()));
        dst_bufs.push(Some(bufs[num_threads + tid].clone()));
        //mrg_bufs.push(Some(bufs[tid + half_r].clone()));
      } else {
        src_bufs.push(None);
        dst_bufs.push(None);
        //mrg_bufs.push(None);
      }
    }
    assert_eq!(src_bufs.len(), dst_bufs.len());
    DeviceReduceWorker{
      tid:  tid,
      nths: num_threads,
      barrier:  barrier,
      width:    width,
      depth:    depth,
      src_bufs: src_bufs,
      dst_bufs: dst_bufs,
      //mrg_bufs: mrg_bufs,
    }
  }

  pub fn process(&mut self, input: &mut DeviceBufferRef<T>, ctx: &DeviceCtxRef) {
    let tid = self.tid;
    let depth = self.depth;
    let &mut DeviceReduceWorker{
      ref mut src_bufs, ref mut dst_bufs, /*ref mut mrg_bufs,*/ .. } = self;

    ctx.sync();
    self.barrier.wait();

    if depth >= 1 {
      assert!(src_bufs[0].is_some());

      let src_buf = src_bufs[0].as_ref().unwrap();
      input.raw_send(src_buf, ctx);
    }

    for d in (0 .. depth) {
      let r = 1 << (d + 1);
      let half_r = r / 2;

      if (tid % r) == half_r {
        assert!(src_bufs[d].is_some());
        assert!(dst_bufs[d].is_some());

        let src_buf = src_bufs[d].as_ref().unwrap();
        let dst_buf = dst_bufs[d].as_ref().unwrap();
        src_buf.send(dst_buf, ctx);
        ctx.sync();
      }
      self.barrier.wait();

      if (tid % r) == 0 {
        assert!(src_bufs[d].is_some());
        assert!(dst_bufs[d].is_some());
        //assert!(mrg_bufs[d].is_some());

        let src_buf = src_bufs[d].as_ref().unwrap();
        let dst_buf = dst_bufs[d].as_ref().unwrap();
        //let mrg_buf = mrg_bufs[d].as_ref().unwrap();
        //mrg_buf.send(dst_buf, ctx);
        ReduceOp::reduce(dst_buf, src_buf, ctx);
        ctx.sync();
      }
      self.barrier.wait();
    }

    /*if tid == 0 {
      assert!(src_bufs[0].is_some());

      let src_buf = src_bufs[0].as_ref().unwrap();
      unsafe { array_cuda_map_print_i32(
          src_buf.as_ptr() as *const i32, src_buf.len() as c_int,
          ctx.stream.ptr,
      ) };
    }*/
  }
}

pub struct DeviceAllReduceWorker<T> where T: Copy + ReduceOp {
  tid:  usize,
  nths: usize,
  barrier:  Arc<Barrier>,
  width:    usize,
  depth:    usize,
  src_bufs: Vec<Option<Arc<RawDeviceBuffer<T>>>>,
  dst_bufs: Vec<Option<Arc<RawDeviceBuffer<T>>>>,
  bc_bufs:  Vec<Option<Arc<RawDeviceBuffer<T>>>>,
}

impl<T> DeviceAllReduceWorker<T> where T: Copy + ReduceOp {
  pub fn new(tid: usize, num_threads: usize, barrier: Arc<Barrier>, bufs: &[Arc<RawDeviceBuffer<T>>]) -> DeviceAllReduceWorker<T> {
    assert_eq!(2 * num_threads, bufs.len());
    let width = num_threads.checked_next_power_of_two().unwrap();
    let depth = width.trailing_zeros() as usize;
    let mut src_bufs = vec![];
    let mut dst_bufs = vec![];
    let mut bc_bufs = vec![];
    // TODO(20151212): permute the buffers so that peer access enabled pairs are
    // spatially adjacent.
    for d in (0 .. depth) {
      let r = 1 << (d + 1);
      let half_r = r / 2;
      if (tid % r) == half_r {
        src_bufs.push(Some(bufs[tid].clone()));
        dst_bufs.push(Some(bufs[num_threads + tid - half_r].clone()));
        bc_bufs.push(None);
      } else if (tid % r) == 0 {
        src_bufs.push(Some(bufs[tid].clone()));
        dst_bufs.push(Some(bufs[num_threads + tid].clone()));
        bc_bufs.push(Some(bufs[tid + half_r].clone()));
      } else {
        src_bufs.push(None);
        dst_bufs.push(None);
        bc_bufs.push(None);
      }
    }
    assert_eq!(src_bufs.len(), dst_bufs.len());
    DeviceAllReduceWorker{
      tid:  tid,
      nths: num_threads,
      barrier:  barrier,
      width:    width,
      depth:    depth,
      src_bufs: src_bufs,
      dst_bufs: dst_bufs,
      bc_bufs:  bc_bufs,
    }
  }

  pub fn process(&mut self, input: &mut DeviceBufferRef<T>, ctx: &DeviceCtxRef) {
    let tid = self.tid;
    let depth = self.depth;
    let &mut DeviceAllReduceWorker{
      ref mut src_bufs, ref mut dst_bufs, ref mut bc_bufs, .. } = self;

    ctx.sync();
    self.barrier.wait();

    if depth >= 1 {
      assert!(src_bufs[0].is_some());

      let src_buf = src_bufs[0].as_ref().unwrap();
      input.raw_send(src_buf, ctx);
    }

    for d in (0 .. depth) {
      let r = 1 << (d + 1);
      let half_r = r / 2;

      if (tid % r) == half_r {
        assert!(src_bufs[d].is_some());
        assert!(dst_bufs[d].is_some());

        let src_buf = src_bufs[d].as_ref().unwrap();
        let dst_buf = dst_bufs[d].as_ref().unwrap();
        src_buf.send(dst_buf, ctx);
        ctx.sync();
      }
      self.barrier.wait();

      if (tid % r) == 0 {
        assert!(src_bufs[d].is_some());
        assert!(dst_bufs[d].is_some());

        let src_buf = src_bufs[d].as_ref().unwrap();
        let dst_buf = dst_bufs[d].as_ref().unwrap();
        ReduceOp::reduce(dst_buf, src_buf, ctx);
        ctx.sync();
      }
      self.barrier.wait();
    }

    for d in (0 .. depth).rev() {
      let r = 1 << (d + 1);
      let half_r = r / 2;

      if (tid % r) == 0 {
        let src_buf = src_bufs[d].as_ref().unwrap();
        let bc_buf = bc_bufs[d].as_ref().unwrap();
        src_buf.send(bc_buf, ctx);
        ctx.sync();
      }
      self.barrier.wait();
    }
  }
}

