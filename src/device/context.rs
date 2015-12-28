//use cuda::ffi::runtime::{cudaDeviceFlags};
use cuda::runtime::{CudaDevice, CudaStream, CudaEvent, /*CudaEventStatus*/};
use cuda_blas::{CublasHandle};
use cuda_dnn::v3::{CudnnHandle};
use cuda_rand::{CurandGenerator};
use cuda_sparse::{CusparseHandle};

use rand::{Rng, thread_rng};
use std::cell::{RefCell, Ref};
use std::ops::{Deref};
use std::sync::{Once, ONCE_INIT};

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

/*pub struct DeviceContext {
  dev_idx:    usize,
  dev_sync:   CudaEvent,

  pub stream: CudaStream,
  pub blas:   CublasHandle,
  pub dnn:    CudnnHandle,
  pub rng:    CurandGenerator,
  pub sparse: CusparseHandle,
}

impl DeviceContext {
  pub fn new(dev_idx: usize) -> DeviceContext {
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

    DeviceContext{
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
}*/

pub type DeviceContext = LazyDeviceContext;

pub struct LazyDeviceContext {
  dev_idx:      usize,
  dev_sync:     CudaEvent,
  pub stream:   CudaStream,

  blas:         RefCell<Option<CublasHandle>>,
  dnn:          RefCell<Option<CudnnHandle>>,
  rng:          RefCell<Option<CurandGenerator>>,
  sparse:       RefCell<Option<CusparseHandle>>,
}

impl LazyDeviceContext {
  pub fn new(dev_idx: usize) -> LazyDeviceContext {
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

    let stream = CudaStream::create()
      .ok().expect("failed to create cuda stream!");

    let dev_sync = CudaEvent::create_fastest()
      .ok().expect("failed to create blocking sync event!");

    /*let blas = CublasHandle::create()
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
      .ok().expect("failed to set stream for cusparse handle!");*/

    LazyDeviceContext{
      dev_idx:  dev_idx,
      dev_sync: dev_sync,
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
    DeviceCtxRef{ctx: self}
  }
}

pub struct DeviceCtxRef<'ctx> {
  ctx:  &'ctx LazyDeviceContext,
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
}
