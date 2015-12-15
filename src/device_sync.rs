use context::{DeviceCtxRef};

use cuda::runtime::{OwnedCudaEvent, SharedCudaEvent};

use std::sync::{Arc};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::usize;

pub fn device_condvar_channel(ctx: &DeviceCtxRef) -> (DeviceCondvarSource, DeviceCondvarSink) {
  let src_epoch = Arc::new(AtomicUsize::new(0));
  let sink_epoch = Arc::new(AtomicUsize::new(0));
  let event = OwnedCudaEvent::create_fastest().unwrap();
  let shared_event = event.share();
  (
    DeviceCondvarSource{
      epoch:        src_epoch.clone(),
      sink_epoch:   sink_epoch.clone(),
      event:        event,
    },
    DeviceCondvarSink{
      epoch:        sink_epoch,
      src_epoch:    src_epoch,
      event:        shared_event,
    },
  )
}

pub struct DeviceCondvarSource {
  epoch:        Arc<AtomicUsize>,
  sink_epoch:   Arc<AtomicUsize>,
  event:        OwnedCudaEvent,
}

impl DeviceCondvarSource {
  pub fn notify(&self, ctx: &DeviceCtxRef) {
    let epoch = self.epoch.load(Ordering::SeqCst);
    while epoch > self.sink_epoch.load(Ordering::SeqCst) {
      // Spin-wait.
    }
    self.event.record(&ctx.stream).unwrap();
    let prev_epoch = self.epoch.fetch_add(1, Ordering::SeqCst);
    assert!(prev_epoch != usize::MAX);
  }
}

pub struct DeviceCondvarSink {
  epoch:        Arc<AtomicUsize>,
  src_epoch:    Arc<AtomicUsize>,
  event:        SharedCudaEvent,
}

impl DeviceCondvarSink {
  pub fn wait(&self, ctx: &DeviceCtxRef) {
    let epoch = self.epoch.load(Ordering::SeqCst);
    while epoch == self.src_epoch.load(Ordering::SeqCst) {
      // Spin-wait.
    }
    ctx.stream.wait_shared_event(&self.event).unwrap();
    let prev_epoch = self.epoch.fetch_add(1, Ordering::SeqCst);
    assert!(prev_epoch != usize::MAX);
  }
}
