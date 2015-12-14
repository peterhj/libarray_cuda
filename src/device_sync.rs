use context::{DeviceCtxRef};

use cuda::runtime::{OwnedCudaEvent, SharedCudaEvent};

use std::sync::{Arc};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct DeviceCondVarSource {
  cap_sinks:  usize,
  num_sinks:  usize,
  epoch:  Arc<AtomicUsize>,
  event:  OwnedCudaEvent,
}

impl DeviceCondVarSource {
  pub fn new(ctx: &DeviceCtxRef) -> DeviceCondVarSource {
    DeviceCondVarSource{
      cap_sinks:  1,
      num_sinks:  0,
      epoch:  Arc::new(AtomicUsize::new(0)),
      event:  OwnedCudaEvent::create_fastest().unwrap(),
    }
  }

  pub fn make_sink(&mut self) -> DeviceCondVarSink {
    assert!(self.num_sinks < self.cap_sinks);
    self.num_sinks += 1;
    DeviceCondVarSink{
      src_ep: self.epoch.clone(),
      epoch:  0,
      event:  self.event.share(),
    }
  }

  pub fn notify(&self, ctx: &DeviceCtxRef) {
    self.event.record(&ctx.stream).unwrap();
    self.epoch.fetch_add(1, Ordering::SeqCst);
  }
}

pub struct DeviceCondVarSink {
  src_ep: Arc<AtomicUsize>,
  epoch:  usize,
  event:  SharedCudaEvent,
}

impl DeviceCondVarSink {
  pub fn wait(&mut self, ctx: &DeviceCtxRef) {
    while self.epoch >= self.src_ep.load(Ordering::SeqCst) {
      // Spin-wait.
    }
    ctx.stream.wait_shared_event(&self.event).unwrap();
    self.epoch += 1;
  }
}
