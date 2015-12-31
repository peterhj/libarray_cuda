use device::context::{
  DeviceContext, DeviceCtxRef,
  DeviceCtxEventProducer, DeviceCtxEventConsumer,
};
use device::ext::*;
use device::linalg::*;
use device::memory::{
  DeviceBufferRef, DeviceBufferRefMut, RawDeviceBuffer,
};
use ffi::*;

//use array_new::{ArrayZeroExt};
use cuda::runtime::{CudaDevice};
use float::stub::{f16_stub};

use libc::{c_int};
use std::cmp::{min};
use std::sync::{Arc};

pub mod allreduce;

pub trait DownsampleOp<U> where U: Copy {
  fn raw_downsample(&self, dst: &RawDeviceBuffer<U>);
}

impl<'ctx> DownsampleOp<f32> for DeviceBufferRef<'ctx, f32> {
  fn raw_downsample(&self, dst: &RawDeviceBuffer<f32>) {
    self.raw_send(dst, &self.ctx);
  }
}

impl<'ctx> DownsampleOp<f16_stub> for DeviceBufferRef<'ctx, f32> {
  fn raw_downsample(&self, dst: &RawDeviceBuffer<f16_stub>) {
    assert_eq!(self.len(), dst.len());
    unsafe { array_cuda_map_cast_f32_to_f16(
        self.as_ptr(), self.len() as c_int,
        dst.as_mut_ptr(),
        self.ctx.stream.ptr,
    ) };
  }
}

pub trait UpsampleOp<U> where U: Copy {
  fn raw_upsample(&mut self, src: &RawDeviceBuffer<U>);
}

impl<'ctx> UpsampleOp<f32> for DeviceBufferRefMut<'ctx, f32> {
  fn raw_upsample(&mut self, src: &RawDeviceBuffer<f32>) {
    assert_eq!(self.len(), src.len());
    self.raw_recv(src);
  }
}

impl<'ctx> UpsampleOp<f16_stub> for DeviceBufferRefMut<'ctx, f32> {
  fn raw_upsample(&mut self, src: &RawDeviceBuffer<f16_stub>) {
    assert_eq!(self.len(), src.len());
    unsafe { array_cuda_map_cast_f16_to_f32(
        src.as_ptr(), src.len() as c_int,
        self.as_mut_ptr(),
        self.ctx.stream.ptr,
    ) };
  }
}

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
    /*unsafe { array_cuda_map_add_f32(
        src.as_ptr(), src.len() as c_int,
        dst.as_mut_ptr(),
        ctx.stream.ptr,
    ) };*/
    dst.async_vector_add(1.0, src, ctx);
  }
}

pub fn for_all_devices<F, V>(limit: usize, mut f: F) -> V where F: FnMut(&[DeviceContext]) -> V {
  let mut contexts = vec![];
  for dev_idx in (0 .. min(limit, CudaDevice::count().unwrap())) {
    let context = DeviceContext::new(dev_idx);
    contexts.push(context);
  }
  f(&contexts)
}

pub struct SendChanSource<T> where T: Copy {
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
    src.raw_send(&self.buf, &src.ctx);
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
    dst.raw_recv(&self.buf);
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
}