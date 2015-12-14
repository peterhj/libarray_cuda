use context::{DeviceContext, DeviceCtxRef};
use device_ext::*;
use device_memory::{
  DeviceBufferRef, RawDeviceBuffer,
  SyncDeviceBuffer, SyncDeviceBufferRef, SyncDeviceBufferRefMut,
};
use device_sync::{DeviceCondVarSource, DeviceCondVarSink};
use ffi::*;

use cuda::runtime::{CudaEvent};

use libc::{c_int};
use std::sync::{Arc, Barrier, Mutex, RwLock};
use vec_map::{VecMap};

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

pub struct DeviceAllReduceSharedData<T> where T: Copy + ReduceOp {
  nths:     usize,
  barrier:  Arc<Barrier>,
  //syncs:    Arc<RwLock<VecMap<Arc<CudaEvent>>>>,
  cv_sinks: Arc<RwLock<VecMap<Arc<Mutex<DeviceCondVarSink>>>>>,
  bufs:     Vec<Arc<RawDeviceBuffer<T>>>,
}

impl<T> DeviceAllReduceSharedData<T> where T: Copy + ReduceOp {
  pub fn new(num_threads: usize, buf_size: usize, ctxs: &[DeviceContext]) -> DeviceAllReduceSharedData<T> {
    let mut bufs = vec![];
    for i in (0 .. 2 * num_threads) {
      let ctx = ctxs[i % num_threads].as_ref();
      bufs.push(Arc::new(RawDeviceBuffer::new(buf_size, &ctx)));
    }
    DeviceAllReduceSharedData{
      nths:     num_threads,
      barrier:  Arc::new(Barrier::new(num_threads)),
      //syncs:    Arc::new(RwLock::new(VecMap::with_capacity(num_threads))),
      cv_sinks: Arc::new(RwLock::new(VecMap::with_capacity(num_threads))),
      bufs:     bufs,
    }
  }
}

pub struct DeviceAllReduceWorker<T> where T: Copy + ReduceOp {
  tid:      usize,
  nths:     usize,
  width:    usize,
  depth:    usize,
  barrier:  Arc<Barrier>,
  //syncs:    Arc<RwLock<VecMap<Arc<CudaEvent>>>>,
  cv_src:   DeviceCondVarSource,
  cv_sinks: Arc<RwLock<VecMap<Arc<Mutex<DeviceCondVarSink>>>>>,
  src_bufs: Vec<Option<Arc<RawDeviceBuffer<T>>>>,
  rd_bufs:  Vec<Option<Arc<RawDeviceBuffer<T>>>>,
  bc_bufs:  Vec<Option<Arc<RawDeviceBuffer<T>>>>,
}

impl<T> DeviceAllReduceWorker<T> where T: Copy + ReduceOp {
  pub fn new(tid: usize, shared: Arc<DeviceAllReduceSharedData<T>>, ctx: &DeviceCtxRef) -> DeviceAllReduceWorker<T> {
    let num_threads = shared.nths;
    assert_eq!(2 * num_threads, shared.bufs.len());
    let width = num_threads.checked_next_power_of_two().unwrap();
    let depth = width.trailing_zeros() as usize;
    let mut src_bufs = vec![];
    let mut rd_bufs = vec![];
    let mut bc_bufs = vec![];
    // TODO(20151212): permute the buffers so that peer access enabled pairs are
    // spatially adjacent.
    for d in (0 .. depth) {
      let r = 1 << (d + 1);
      let half_r = r / 2;
      if (tid % r) == half_r {
        src_bufs.push(Some(shared.bufs[tid].clone()));
        rd_bufs.push(Some(shared.bufs[num_threads + tid - half_r].clone()));
        bc_bufs.push(None);
      } else if (tid % r) == 0 {
        src_bufs.push(Some(shared.bufs[tid].clone()));
        rd_bufs.push(Some(shared.bufs[num_threads + tid].clone()));
        bc_bufs.push(Some(shared.bufs[tid + half_r].clone()));
      } else {
        src_bufs.push(None);
        rd_bufs.push(None);
        bc_bufs.push(None);
      }
    }
    assert_eq!(src_bufs.len(), rd_bufs.len());
    assert_eq!(src_bufs.len(), bc_bufs.len());
    /*let mut syncs = shared.syncs.clone();
    {
      let mut syncs = syncs.write().unwrap();
      syncs.insert(tid, Arc::new(CudaEvent::create_with_flags(0x02).unwrap()));
    }*/
    let mut cv_src = DeviceCondVarSource::new(ctx);
    let mut cv_sinks = shared.cv_sinks.clone();
    {
      let mut cv_sinks = cv_sinks.write().unwrap();
      cv_sinks.insert(tid, Arc::new(Mutex::new(cv_src.make_sink())));
    }
    DeviceAllReduceWorker{
      tid:      tid,
      nths:     num_threads,
      width:    width,
      depth:    depth,
      barrier:  shared.barrier.clone(),
      //syncs:    syncs,
      cv_src:   cv_src,
      cv_sinks: cv_sinks,
      src_bufs: src_bufs,
      rd_bufs:  rd_bufs,
      bc_bufs:  bc_bufs,
    }
  }

  pub fn process(&mut self, input: &mut DeviceBufferRef<T>, ctx: &DeviceCtxRef) {
    let tid = self.tid;
    let depth = self.depth;
    let &mut DeviceAllReduceWorker{
      ref mut src_bufs, ref mut rd_bufs, ref mut bc_bufs, .. } = self;

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
        assert!(rd_bufs[d].is_some());

        let src_buf = src_bufs[d].as_ref().unwrap();
        let rd_buf = rd_bufs[d].as_ref().unwrap();
        src_buf.send(rd_buf, ctx);
        ctx.sync();
      }
      self.barrier.wait();

      if (tid % r) == 0 {
        assert!(src_bufs[d].is_some());
        assert!(rd_bufs[d].is_some());

        let src_buf = src_bufs[d].as_ref().unwrap();
        let rd_buf = rd_bufs[d].as_ref().unwrap();
        ReduceOp::reduce(rd_buf, src_buf, ctx);
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

