use device::context::{
  DeviceContext, DeviceCtxRef,
  DeviceCtxEventProducer, DeviceCtxEventConsumer,
};
use device::ext::*;
use device::linalg::*;
use device::memory::{
  DeviceBufferRef, DeviceBufferRefMut, RawDeviceBufferRef,
};
use ffi::*;

//use array_new::{ArrayZeroExt};
use cuda::runtime::{CudaDevice};
use float::stub::{f16_stub};

use libc::{c_int};
use std::cmp::{min};
use std::marker::{PhantomData};
use std::sync::{Arc};

pub mod allreduce;

pub trait DownsampleOp<U> where U: Copy {
  fn raw_downsample(&self, dst: &RawDeviceBufferRef<U>);
}

impl<'ctx> DownsampleOp<f32> for DeviceBufferRef<'ctx, f32> {
  fn raw_downsample(&self, dst: &RawDeviceBufferRef<f32>) {
    self.raw_send(dst);
  }
}

impl<'ctx> DownsampleOp<f16_stub> for DeviceBufferRef<'ctx, f32> {
  fn raw_downsample(&self, dst: &RawDeviceBufferRef<f16_stub>) {
    assert_eq!(self.len(), dst.len());
    unsafe { array_cuda_map_cast_f32_to_f16(
        self.as_ptr(), self.len() as c_int,
        dst.as_mut_ptr(),
        self.ctx.stream.ptr,
    ) };
  }
}

pub trait UpsampleOp<U> where U: Copy {
  fn raw_upsample(&mut self, src: &RawDeviceBufferRef<U>);
}

impl<'ctx> UpsampleOp<f32> for DeviceBufferRefMut<'ctx, f32> {
  fn raw_upsample(&mut self, src: &RawDeviceBufferRef<f32>) {
    assert_eq!(self.len(), src.len());
    self.raw_recv(src);
  }
}

impl<'ctx> UpsampleOp<f16_stub> for DeviceBufferRefMut<'ctx, f32> {
  fn raw_upsample(&mut self, src: &RawDeviceBufferRef<f16_stub>) {
    assert_eq!(self.len(), src.len());
    unsafe { array_cuda_map_cast_f16_to_f32(
        src.as_ptr(), src.len() as c_int,
        self.as_mut_ptr(),
        self.ctx.stream.ptr,
    ) };
  }
}

pub trait ReduceOp {
  fn reduce<'ctx>(src: &RawDeviceBufferRef<Self>, dst: &RawDeviceBufferRef<Self>, ctx: &DeviceCtxRef<'ctx>) where Self: Copy;
}

impl ReduceOp for i32 {
  fn reduce<'ctx>(src: &RawDeviceBufferRef<i32>, dst: &RawDeviceBufferRef<i32>, ctx: &DeviceCtxRef<'ctx>) {
    unsafe { array_cuda_map_add_i32(
        src.as_ptr(), src.len() as c_int,
        dst.as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }
}

impl ReduceOp for f32 {
  fn reduce<'ctx>(src: &RawDeviceBufferRef<f32>, dst: &RawDeviceBufferRef<f32>, ctx: &DeviceCtxRef<'ctx>) {
    //dst.async_vector_add(1.0, src, ctx);
    unsafe { array_cuda_map_add_f32(
        1.0, src.as_ptr(), src.len() as c_int,
        1.0, dst.as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }
}

pub fn for_all_devices<F, V>(limit: usize, mut f: F) -> V where F: FnMut(&[DeviceContext]) -> V {
  let mut contexts = vec![];
  // XXX(20160320): Cycle between all devices in round-robin order.
  //for dev_idx in 0 .. min(limit, CudaDevice::count().unwrap()) {
  for worker_idx in 0 .. limit {
    let context = DeviceContext::new(worker_idx);
    contexts.push(context);
  }
  f(&contexts)
}

pub trait ReduceOperation<T> where T: Copy {
  fn reduce<'ctx>(&self, src: &RawDeviceBufferRef<T>, dst: &RawDeviceBufferRef<T>, ctx: &DeviceCtxRef<'ctx>);
}

pub struct SumReduceOperation<T> where T: Copy {
  _marker:      PhantomData<T>,
}

impl<T> SumReduceOperation<T> where T: Copy {
  pub fn new() -> SumReduceOperation<T> {
    SumReduceOperation{
      _marker:      PhantomData,
    }
  }
}

impl ReduceOperation<f32> for SumReduceOperation<f32> {
  fn reduce<'ctx>(&self, src: &RawDeviceBufferRef<f32>, dst: &RawDeviceBufferRef<f32>, ctx: &DeviceCtxRef<'ctx>) {
    unsafe { array_cuda_map_add_f32(
        1.0, src.as_ptr(), src.len() as c_int,
        1.0, dst.as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }
}

pub struct AverageReduceOperation<T> where T: Copy {
  num_workers:  usize,
  _marker:      PhantomData<T>,
}

impl<T> AverageReduceOperation<T> where T: Copy {
  pub fn new(num_workers: usize) -> AverageReduceOperation<T> {
    AverageReduceOperation{
      num_workers:  num_workers,
      _marker:      PhantomData,
    }
  }
}

impl ReduceOperation<f32> for AverageReduceOperation<f32> {
  fn reduce<'ctx>(&self, src: &RawDeviceBufferRef<f32>, dst: &RawDeviceBufferRef<f32>, ctx: &DeviceCtxRef<'ctx>) {
    unsafe { array_cuda_map_add_f32(
        0.5, src.as_ptr(), src.len() as c_int,
        0.5, dst.as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }
}

/*pub struct SendChanSource<T> where T: Copy {
  send_ev:  DeviceCtxEventProducer,
  recv_ev:  DeviceCtxEventConsumer,
  buf:      Arc<RawDeviceBuffer<T>>,
  init:     bool,
}

impl<T> SendChanSource<T> where T: Copy {
  pub fn send(&mut self, src: &DeviceBufferRef<T>) {
    assert_eq!(self.send_ev.device(), src.ctx.device());
    if !self.init {
      self.init = true;
    } else {
      self.recv_ev.consume(&src.ctx);
    }
    src.raw_send(&self.buf.as_ref());
    self.send_ev.produce(&src.ctx);
  }
}

pub struct SendChanSink<T> where T: Copy {
  send_ev:  DeviceCtxEventConsumer,
  recv_ev:  DeviceCtxEventProducer,
  buf:      Arc<RawDeviceBuffer<T>>,
}

impl<T> SendChanSink<T> where T: Copy {
  pub fn recv(&self, dst: &mut DeviceBufferRefMut<T>) {
    assert_eq!(self.recv_ev.device(), dst.ctx.device());
    self.send_ev.consume(&dst.ctx);
    dst.raw_recv(&self.buf.as_ref());
    self.recv_ev.produce(&dst.ctx);
  }
}

pub fn send_channel<T: Copy>(buf_len: usize, send_ctx: &DeviceContext, recv_ctx: &DeviceContext) -> (SendChanSource<T>, SendChanSink<T>) {
  let (send_tx, send_rx) = {
    let ctx = send_ctx.as_ref();
    DeviceCtxEventProducer::channel(&ctx)
  };
  let ((recv_tx, recv_rx), buf) = {
    let ctx = recv_ctx.as_ref();
    (DeviceCtxEventProducer::channel(&ctx), Arc::new(unsafe { RawDeviceBuffer::new(buf_len, &ctx) }))
  };
  (SendChanSource{
    send_ev:    send_tx,
    recv_ev:    recv_rx,
    buf:        buf.clone(),
    init:       false,
  },
  SendChanSink{
    send_ev:    send_rx,
    recv_ev:    recv_tx,
    buf:        buf,
  })
}*/
