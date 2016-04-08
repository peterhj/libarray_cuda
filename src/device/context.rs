//use cuda::ffi::runtime::{cudaDeviceFlags};
use cuda::runtime::{
  CudaDevice, CudaStream, CudaEvent, /*CudaEventStatus*/
  OwnedCudaEvent, SharedCudaEvent,
};
use cuda_blas::{CublasHandle};
use cuda_dnn::v4::{CudnnHandle};
use cuda_rand::{CurandGenerator};
use cuda_sparse::{CusparseHandle};

use rand::{Rng, thread_rng};
use std::cell::{RefCell, Ref};
use std::ops::{Deref};
use std::rc::{Rc};
use std::sync::{Arc, Once, ONCE_INIT};
use std::sync::atomic::{AtomicUsize, Ordering};

static DEVICE_CONTEXT_INIT: Once = ONCE_INIT;

#[derive(Clone, Copy)]
enum EventState {
  Produce,
  Consume,
}

pub struct DeviceCtxEvent {
  dev_idx:    usize,
  dev_sync:   CudaEvent,
  state:      EventState,
}

impl DeviceCtxEvent {
  pub fn new(ctx: &DeviceCtxRef) -> DeviceCtxEvent {
    DeviceCtxEvent{
      dev_idx:  ctx.device(),
      dev_sync: CudaEvent::create_fastest().unwrap(),
      state:    EventState::Produce,
    }
  }

  pub fn produce(&mut self, ctx: &DeviceCtxRef) {
    assert_eq!(self.dev_idx, ctx.device());
    match self.state {
      EventState::Produce => {
        self.dev_sync.record(&ctx.stream).unwrap();
        self.state = EventState::Consume;
      }
      EventState::Consume => {
        panic!("DeviceCtxEvent::produce() in bad state");
      }
    }
  }

  pub fn consume(&mut self, ctx: &DeviceCtxRef) {
    match self.state {
      EventState::Produce => {
        panic!("DeviceCtxEvent::consume() in bad state");
      }
      EventState::Consume => {
        ctx.stream.wait_event(&self.dev_sync).unwrap();
        self.state = EventState::Produce;
      }
    }
  }
}

/*pub struct DeviceCtxSharedEvent {
  dev_idx:    usize,
  dev_sync:   CudaEvent,
  state:      AtomicUsize,
}

impl DeviceCtxSharedEvent {
  pub fn new(ctx: &DeviceCtxRef) -> DeviceCtxSharedEvent {
    DeviceCtxSharedEvent{
      dev_idx:  ctx.device(),
      dev_sync: CudaEvent::create_fastest().unwrap(),
      state:    AtomicUsize::new(0),
    }
  }

  pub fn produce(&mut self, ctx: &DeviceCtxRef) {
    assert_eq!(self.dev_idx, ctx.device());
    loop {
      match self.state.load(Ordering::Acquire) {
        0 => {
          self.dev_sync.record(&ctx.stream).unwrap();
          self.state.store(1, Ordering::Release);
          break;
        }
        1 => {}
        _ => { unreachable!(); }
      }
    }
  }

  pub fn consume(&mut self, ctx: &DeviceCtxRef) {
    loop {
      match self.state.load(Ordering::Acquire) {
        0 => {}
        1 => {
          ctx.stream.wait_event(&self.dev_sync).unwrap();
          self.state.store(0, Ordering::Release);
        }
        _ => { unreachable!(); }
      }
    }
  }
}*/

pub struct DeviceCtxEventProducer {
  dev_idx:      usize,
  dev_event:    OwnedCudaEvent,
  state:        Arc<AtomicUsize>,
}

impl DeviceCtxEventProducer {
  pub fn channel(ctx: &DeviceCtxRef) -> (DeviceCtxEventProducer, DeviceCtxEventConsumer) {
    let dev_idx = ctx.device();
    let owned_event = OwnedCudaEvent::create_fastest().unwrap();
    let shared_event = owned_event.share();
    let state = Arc::new(AtomicUsize::new(0));
    (DeviceCtxEventProducer{
      dev_idx:      dev_idx,
      dev_event:    owned_event,
      state:        state.clone(),
    },
    DeviceCtxEventConsumer{
      dev_idx:      dev_idx,
      dev_event:    shared_event,
      state:        state,
    })
  }

  pub fn device(&self) -> usize {
    self.dev_idx
  }

  pub fn try_produce(&self, ctx: &DeviceCtxRef) -> Result<(), ()> {
    assert_eq!(self.dev_idx, ctx.device());
    let state = self.state.load(Ordering::Acquire);
    match state {
      0 => {
        self.dev_event.record(&ctx.stream).unwrap();
        self.state.store(1, Ordering::Release);
        Ok(())
      }
      1 => {
        Err(())
      }
      _ => unreachable!(),
    }
  }

  pub fn produce(&self, ctx: &DeviceCtxRef) {
    while self.try_produce(ctx).is_err() {
    }
  }
}

pub struct DeviceCtxEventConsumer {
  dev_idx:      usize,
  dev_event:    SharedCudaEvent,
  state:        Arc<AtomicUsize>,
}

impl DeviceCtxEventConsumer {
  pub fn try_consume(&self, ctx: &DeviceCtxRef) -> Result<(), ()> {
    let state = self.state.load(Ordering::Acquire);
    match state {
      0 => {
        Err(())
      }
      1 => {
        ctx.stream.wait_shared_event(&self.dev_event).unwrap();
        self.state.store(0, Ordering::Release);
        Ok(())
      }
      _ => unreachable!(),
    }
  }

  pub fn consume(&self, ctx: &DeviceCtxRef) {
    while self.try_consume(ctx).is_err() {
    }
  }
}

pub struct OldDeviceContext {
  dev_idx:    usize,
  dev_sync:   CudaEvent,

  pub stream: CudaStream,
  pub blas:   CublasHandle,
  pub dnn:    CudnnHandle,
  pub rng:    CurandGenerator,
  pub sparse: CusparseHandle,
}

impl OldDeviceContext {
  pub fn new(dev_idx: usize) -> OldDeviceContext {
    DEVICE_CONTEXT_INIT.call_once(|| {
      // TODO(20151211): see notes in cuda docs:
      // <http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html>
      let mut peer_pairs = vec![];
      for dst in CudaDevice::iter().unwrap() {
        match CudaDevice::set_current(dst) {
          Ok(_) => {}
          Err(e) => {
            panic!("failed to set current device ({}): {:?}", dev_idx, e);
          }
        }
        for src in CudaDevice::iter().unwrap() {
          if dst == src {
            continue;
          }
          if CudaDevice::can_access_peer(dst, src).unwrap() {
            CudaDevice::enable_peer_access(src).unwrap();
            peer_pairs.push((src, dst));
          }
        }
      }
      println!("DEBUG: cuda peer access pairs: {:?}", peer_pairs);
    });

    match CudaDevice::set_current(dev_idx) {
      Ok(_) => {}
      Err(e) => {
        panic!("failed to set current device ({}): {:?}", dev_idx, e);
      }
    }

    //let stream = CudaStream::default();
    let stream = CudaStream::create()
      .ok().expect("failed to create cuda stream!");

    //let dev_sync = CudaEvent::create_with_flags(0x02)
    let dev_sync = CudaEvent::create_fastest()
      .ok().expect("failed to create blocking sync event!");

    let blas = CublasHandle::create()
      .ok().expect("failed to create cublas handle!");
    blas.set_stream(&stream)
      .ok().expect("failed to set stream for cublas handle!");

    let dnn = CudnnHandle::create()
      .ok().expect("failed to create cudnn handle!");
    dnn.set_stream(&stream)
      .ok().expect("failed to set stream for cudnn handle!");

    let rng = CurandGenerator::create()
      .ok().expect("failed to create curand generator!");
    rng.set_stream(&stream)
      .ok().expect("failed to set stream for curand generator!");
    rng.set_seed(thread_rng().next_u64())
      .ok().expect("failed to set seed for curand generator!");
    rng.set_offset(0)
      .ok().expect("failed to set offset for curand generator!");

    let sparse = CusparseHandle::create()
      .ok().expect("failed to create cusparse handle!");
    sparse.set_stream(&stream)
      .ok().expect("failed to set stream for cusparse handle!");

    OldDeviceContext{
      dev_idx:  dev_idx,
      dev_sync: dev_sync,
      stream: stream,
      blas:   blas,
      dnn:    dnn,
      rng:    rng,
      sparse: sparse,
    }
  }

  pub unsafe fn set_device(&self) {
    match CudaDevice::set_current(self.dev_idx) {
      Ok(_) => {}
      Err(e) => {
        panic!("failed to set current device ({}): {:?}", self.dev_idx, e);
      }
    }
  }

  /*pub fn as_ref<'ctx>(&'ctx self) -> DeviceCtxRef<'ctx> {
    unsafe { self.set_device() };
    DeviceCtxRef{ctx: self}
  }*/
}

// XXX(20151228): DeviceContext is implemented such that the current CUDA device
// is set upon acquiring a DeviceCtxRef. Therefore there should only be at most
// one outstanding DeviceCtxRef in a thread. We can achieve it by giving each
// DeviceCtxRef some sort of exclusive reference to a DriverContext.

thread_local!(static DRIVER_CONTEXT: Rc<DriverContext> = Rc::new(DriverContext));

pub struct DriverContext;

impl !Send for DriverContext {}
impl !Sync for DriverContext {}

pub type DeviceContext = LazyDeviceContext;

pub struct LazyDeviceContext {
  dev_idx:      usize,
  //dev_sync:     CudaEvent,
  pub stream:   CudaStream,

  blas:         RefCell<Option<CublasHandle>>,
  dnn:          RefCell<Option<CudnnHandle>>,
  rng:          RefCell<Option<CurandGenerator>>,
  sparse:       RefCell<Option<CusparseHandle>>,
}

impl LazyDeviceContext {
  pub fn new(worker_idx: usize) -> LazyDeviceContext {
    let dev_count = CudaDevice::count().unwrap();
    let dev_idx = worker_idx % dev_count;
    DEVICE_CONTEXT_INIT.call_once(|| {
      // TODO(20151211): see notes in cuda docs:
      // <http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html>
      let mut peer_pairs = vec![];
      for dst in CudaDevice::iter().unwrap() {
        match CudaDevice::set_current(dst) {
          Ok(_) => {}
          Err(e) => {
            panic!("failed to set current device ({}): {:?}", dev_idx, e);
          }
        }
        for src in CudaDevice::iter().unwrap() {
          if dst == src {
            continue;
          }
          if CudaDevice::can_access_peer(dst, src).unwrap() {
            CudaDevice::enable_peer_access(src).unwrap();
            peer_pairs.push((src, dst));
          }
        }
      }
      //println!("DEBUG: cuda peer access pairs: {:?}", peer_pairs);
    });

    match CudaDevice::set_current(dev_idx) {
      Ok(_) => {}
      Err(e) => {
        panic!("failed to set current device ({}): {:?}", dev_idx, e);
      }
    }

    let stream = CudaStream::create()
      .ok().expect("failed to create cuda stream!");

    /*let dev_sync = CudaEvent::create_fastest()
      .ok().expect("failed to create blocking sync event!");*/

    /*let rng = CurandGenerator::create()
      .ok().expect("failed to create curand generator!");
    rng.set_stream(&stream)
      .ok().expect("failed to set stream for curand generator!");
    rng.set_seed(thread_rng().next_u64())
      .ok().expect("failed to set seed for curand generator!");
    rng.set_offset(0)
      .ok().expect("failed to set offset for curand generator!");

    let sparse = CusparseHandle::create()
      .ok().expect("failed to create cusparse handle!");
    sparse.set_stream(&stream)
      .ok().expect("failed to set stream for cusparse handle!");*/

    LazyDeviceContext{
      dev_idx:  dev_idx,
      //dev_sync: dev_sync,
      stream:   stream,
      blas:     RefCell::new(None),
      dnn:      RefCell::new(None),
      rng:      RefCell::new(None),
      sparse:   RefCell::new(None),
    }
  }

  pub unsafe fn set_device(&self) {
    match CudaDevice::set_current(self.dev_idx) {
      Ok(_) => {}
      Err(e) => {
        panic!("failed to set current device ({}): {:?}", self.dev_idx, e);
      }
    }
  }

  pub fn as_ref<'ctx>(&'ctx self) -> DeviceCtxRef<'ctx> {
    unsafe { self.set_device() };
    //DeviceCtxRef{ctx: self, driver: None}
    DRIVER_CONTEXT.with(|driver| {
      let driver = driver.clone();
      assert!(Rc::strong_count(&driver) <= 2,
          "DeviceCtxRef requires exclusive reference to DriverContext!");
      DeviceCtxRef{ctx: self, driver: driver}
    })
  }
}

pub struct DeviceCtxRef<'ctx> {
  ctx:      &'ctx LazyDeviceContext,
  driver:   Rc<DriverContext>,
}

impl<'ctx> Deref for DeviceCtxRef<'ctx> {
  type Target = LazyDeviceContext;

  fn deref(&self) -> &LazyDeviceContext {
    self.ctx
  }
}

impl<'ctx> DeviceCtxRef<'ctx> {
  /*pub unsafe fn as_raw_ptr(&self) -> *const RawDeviceContext {
    self.ctx.stream.ptr as *const RawDeviceContext
  }*/

  pub fn device(&self) -> usize {
    self.ctx.dev_idx
  }

  pub fn sync(&self) {
    self.ctx.stream.synchronize()
      .ok().expect("failed to synchronize cuda stream!");
  }

  pub fn blocking_sync(&self) {
    // TODO(20151227)
    /*self.ctx.dev_sync.record(&self.stream).unwrap();
    self.ctx.dev_sync.synchronize().unwrap();*/
    self.ctx.stream.synchronize()
      .ok().expect("failed to synchronize cuda stream!");
  }

  pub fn get_blas(&self) -> Ref<CublasHandle> {
    {
      let mut blas = self.ctx.blas.borrow_mut();
      if blas.is_none() {
        let new_blas = CublasHandle::create()
          .ok().expect("failed to create cublas handle!");
        new_blas.set_stream(&self.ctx.stream)
          .ok().expect("failed to set stream for cublas handle!");
        *blas = Some(new_blas);
      }
    }
    let blas: Ref<Option<CublasHandle>> = self.ctx.blas.borrow();
    Ref::map(blas, |h| h.as_ref().unwrap())
  }

  pub fn get_dnn(&self) -> Ref<CudnnHandle> {
    {
      let mut dnn = self.ctx.dnn.borrow_mut();
      if dnn.is_none() {
        let new_dnn = CudnnHandle::create()
          .ok().expect("failed to create cudnn handle!");
        new_dnn.set_stream(&self.ctx.stream)
          .ok().expect("failed to set stream for cudnn handle!");
        *dnn = Some(new_dnn);
      }
    }
    let dnn: Ref<Option<CudnnHandle>> = self.ctx.dnn.borrow();
    Ref::map(dnn, |h| h.as_ref().unwrap())
  }

  pub fn get_rng(&self) -> Ref<CurandGenerator> {
    {
      let mut rng = self.ctx.rng.borrow_mut();
      if rng.is_none() {
        let new_rng = CurandGenerator::create()
          .ok().expect("failed to create curand handle!");
        new_rng.set_stream(&self.ctx.stream)
          .ok().expect("failed to set stream for curand handle!");
        new_rng.set_seed(thread_rng().next_u64())
          .ok().expect("failed to set seed for curand handle!");
        new_rng.set_offset(0)
          .ok().expect("failed to set offset for curand handle!");
        *rng = Some(new_rng);
      }
    }
    let rng: Ref<Option<CurandGenerator>> = self.ctx.rng.borrow();
    Ref::map(rng, |h| h.as_ref().unwrap())
  }
}
