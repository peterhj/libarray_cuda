use device::comm::{ReduceOp};
use device::context::{DeviceContext, DeviceCtxRef};
use device::ext::*;
use device::linalg::*;
use device::memory::{
  DeviceBufferRef, DeviceBufferRefMut, RawDeviceBuffer,
};
//use device::sync::{DeviceCondvarSource, DeviceCondvarSink, device_condvar_channel};
use ffi::*;

use libc::{c_int};
use std::marker::{PhantomData};
use std::sync::{Arc, Barrier, Mutex, RwLock};
use vec_map::{VecMap};

pub struct DeviceAllReduceSharedData<T> where T: Copy + ReduceOp {
  nths:     usize,
  barrier:  Arc<Barrier>,
  //cv_sinks: Arc<RwLock<VecMap<Arc<DeviceCondvarSink>>>>,
  bufs:     Vec<Arc<RawDeviceBuffer<T>>>,
}

impl<T> DeviceAllReduceSharedData<T> where T: Copy + ReduceOp {
  pub fn new(num_threads: usize, buf_size: usize, ctxs: &[DeviceContext]) -> DeviceAllReduceSharedData<T> {
    let mut bufs = vec![];
    for i in 0 .. 2 * num_threads {
      let ctx = ctxs[i % num_threads].as_ref();
      bufs.push(Arc::new(unsafe { RawDeviceBuffer::new(buf_size, &ctx) }));
    }
    DeviceAllReduceSharedData{
      nths:     num_threads,
      barrier:  Arc::new(Barrier::new(num_threads)),
      //cv_sinks: Arc::new(RwLock::new(VecMap::with_capacity(num_threads))),
      bufs:     bufs,
    }
  }
}

pub struct DeviceAllReduceWorker<T, U=T> where T: ReduceOp + Copy, U: ReduceOp + Copy {
  tid:      usize,
  nths:     usize,
  width:    usize,
  depth:    usize,
  pub barrier:  Arc<Barrier>,
  /*cv_src:   DeviceCondvarSource,
  cv_sinks: Arc<RwLock<VecMap<Arc<DeviceCondvarSink>>>>,*/
  src_bufs: Vec<Option<Arc<RawDeviceBuffer<T>>>>,
  rd_bufs:  Vec<Option<Arc<RawDeviceBuffer<T>>>>,
  bc_bufs:  Vec<Option<Arc<RawDeviceBuffer<T>>>>,
  _marker:  PhantomData<U>,
}

impl<T, U> DeviceAllReduceWorker<T, U> where T: ReduceOp + Copy, U: ReduceOp + Copy {
  pub fn new(tid: usize, shared: &DeviceAllReduceSharedData<T>, ctx: &DeviceCtxRef) -> DeviceAllReduceWorker<T> {
    let num_threads = shared.nths;
    assert_eq!(2 * num_threads, shared.bufs.len());
    let width = num_threads.checked_next_power_of_two().unwrap();
    let depth = width.trailing_zeros() as usize;
    let mut src_bufs = vec![];
    let mut rd_bufs = vec![];
    let mut bc_bufs = vec![];
    // TODO(20151212): permute the buffers so that peer access enabled pairs are
    // spatially adjacent.
    for d in 0 .. depth {
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
    /*let (cv_src, cv_sink) = device_condvar_channel(ctx);
    let mut cv_sinks = shared.cv_sinks.clone();
    {
      let mut cv_sinks = cv_sinks.write().unwrap();
      cv_sinks.insert(tid, Arc::new(cv_sink));
    }*/
    DeviceAllReduceWorker{
      tid:      tid,
      nths:     num_threads,
      width:    width,
      depth:    depth,
      barrier:  shared.barrier.clone(),
      /*cv_src:   cv_src,
      cv_sinks: cv_sinks,*/
      src_bufs: src_bufs,
      rd_bufs:  rd_bufs,
      bc_bufs:  bc_bufs,
      _marker:  PhantomData,
    }
  }

  pub fn num_workers(&self) -> usize {
    self.nths
  }

  pub fn load(&mut self, offset: usize, input: &mut DeviceBufferRef<T>) -> usize {
    let &mut DeviceAllReduceWorker{
      ref mut src_bufs, .. } = self;

    assert!(src_bufs[0].is_some());

    let end_offset = offset + input.len();
    let src_buf = src_bufs[0].as_ref().unwrap();
    /*println!("DEBUG: allreduce: load: offsets: {} {} input len: {} src len: {}",
        offset, end_offset,
        input.len(), src_buf.len());*/
    input.raw_send(&src_buf.as_ref_range(offset, end_offset));

    end_offset
  }

  pub fn store(&mut self, offset: usize, output: &mut DeviceBufferRefMut<T>) -> usize {
    let &mut DeviceAllReduceWorker{
      ref mut src_bufs, .. } = self;

    assert!(src_bufs[0].is_some());

    let end_offset = offset + output.len();
    let src_buf = src_bufs[0].as_ref().unwrap();
    output.raw_recv(&src_buf.as_ref_range(offset, end_offset));

    end_offset
  }

  pub fn process(&mut self, input: &mut DeviceBufferRef<T>) {
    {
      let tid = self.tid;
      let depth = self.depth;
      let &mut DeviceAllReduceWorker{
        ref mut src_bufs, ref mut rd_bufs, ref mut bc_bufs,
        //ref cv_src, ref cv_sinks,
        .. } = self;

      if depth >= 1 {
        assert!(src_bufs[0].is_some());

        //input.ctx.sync();
        //self.barrier.wait();

        let src_buf = src_bufs[0].as_ref().unwrap();
        input.raw_send(&(**src_buf).as_ref());
      }
    }

    self.communicate(&input.ctx);
  }

  pub fn communicate(&mut self, ctx: &DeviceCtxRef) {
    let tid = self.tid;
    let depth = self.depth;
    let &mut DeviceAllReduceWorker{
      ref mut src_bufs, ref mut rd_bufs, ref mut bc_bufs,
      .. } = self;

    ctx.sync();
    self.barrier.wait();

    for d in 0 .. depth {
      let r = 1 << (d + 1);
      let half_r = r / 2;

      if (tid % r) == half_r {
        assert!(src_bufs[d].is_some());
        assert!(rd_bufs[d].is_some());

        let src_buf = src_bufs[d].as_ref().unwrap();
        let rd_buf = rd_bufs[d].as_ref().unwrap();
        src_buf.raw_send(rd_buf, ctx);

        //cv_src.notify(ctx);
      }
      ctx.sync();
      self.barrier.wait();

      if (tid % r) == 0 {
        assert!(src_bufs[d].is_some());
        assert!(rd_bufs[d].is_some());

        /*let cv_sinks = cv_sinks.read().unwrap();
        cv_sinks[tid + half_r].*/

        let rd_buf = rd_bufs[d].as_ref().unwrap();
        let src_buf = src_bufs[d].as_ref().unwrap();
        ReduceOp::reduce(&(**rd_buf).as_ref(), &(**src_buf).as_ref(), ctx);

        //cv_src.notify(ctx);
      }
      ctx.sync();
      self.barrier.wait();
    }

    for d in (0 .. depth).rev() {
      let r = 1 << (d + 1);
      let half_r = r / 2;

      if (tid % r) == 0 {
        let src_buf = src_bufs[d].as_ref().unwrap();
        let bc_buf = bc_bufs[d].as_ref().unwrap();
        src_buf.raw_send(bc_buf, ctx);
      }
      ctx.sync();
      self.barrier.wait();
    }

    /*if tid == 0 {
      assert!(src_bufs[0].is_some());
      let src_buf = src_bufs[0].as_ref().unwrap();
      unsafe { array_cuda_map_print_f32(
          src_buf.as_ptr() as *const f32, src_buf.len() as c_int,
          ctx.stream.ptr,
      ) };
    }*/
  }
}
