use device::array::{DeviceArray2dView, DeviceArray2dViewMut};
use device::context::{DeviceCtxRef};
use device::memory::{DeviceBufferRef, DeviceBufferRefMut, RawDeviceBufferRef};
use ffi::*;

use array::{Shape, ArrayView, ArrayViewMut};
use cuda_blas::{
  CublasPointerMode,
  CublasTranspose,
  cublas_sscal,
  cublas_saxpy,
  //cublas_sgemv,
  cublas_sgemm,
};
use cuda_blas::ffi::*;

#[derive(Clone, Copy)]
pub enum Transpose {
  N,
  T,
}

impl Transpose {
  pub fn to_cublas(&self) -> CublasTranspose {
    match self {
      &Transpose::N => CublasTranspose::N,
      &Transpose::T => CublasTranspose::T,
      //&Transpose::H => CublasTranspose::H,
    }
  }
}

pub trait VectorExt {
  type Vector;
  type RawVector;

  fn vector_scale(&mut self, alpha: f32);
  fn vector_add(&mut self, alpha: f32, x: &Self::Vector, beta: f32);
  fn vector_add_raw(&mut self, alpha: f32, x: &Self::RawVector);
  fn vector_elemwise_mult(&mut self, x: &Self::Vector);
}

impl<'a> VectorExt for DeviceBufferRefMut<'a, f32> {
  type Vector = DeviceBufferRef<'a, f32>;
  type RawVector = RawDeviceBufferRef<'a, f32>;

  fn vector_scale(&mut self, alpha: f32) {
    if alpha == 1.0 {
      return;
    }
    let n = self.len();
    unsafe { array_cuda_vector_scale_f32(
        self.as_mut_ptr(),
        n as i32,
        alpha,
        self.ctx.stream.ptr,
    ) };
  }

  fn vector_add(&mut self, alpha: f32, x: &DeviceBufferRef<'a, f32>, beta: f32) {
    let n = self.len();
    let x_n = x.len();
    assert_eq!(n, x_n);
    unsafe { array_cuda_vector_add_f32(
        x.as_ptr(),
        n as i32,
        alpha,
        beta,
        self.as_mut_ptr(),
        self.ctx.stream.ptr,
    ) };
  }

  fn vector_add_raw(&mut self, alpha: f32, x: &RawDeviceBufferRef<'a, f32>) {
    let n = self.len();
    let x_n = x.len();
    assert_eq!(n, x_n);

    self.ctx.get_blas().set_pointer_mode(CublasPointerMode::Host);
    unsafe { cublas_saxpy(
        &*self.ctx.get_blas(),
        n,
        alpha,
        x.as_ptr(), 1,
        self.as_mut_ptr(), 1,
    ) }.unwrap();
  }

  fn vector_elemwise_mult(&mut self, x: &Self::Vector) {
    let n = self.len();
    let x_n = x.len();
    assert_eq!(n, x_n);

    unsafe { array_cuda_vector_elemwise_mult_f32(
        x.as_ptr(),
        n as i32,
        self.as_mut_ptr(),
        self.ctx.stream.ptr,
    ) };
  }
}

pub trait AsyncVectorExt {
  type Ctx;
  type Vector;

  fn async_vector_scale(&mut self, alpha: f32, ctx: &Self::Ctx);
  fn async_vector_add(&mut self, alpha: f32, x: &Self::Vector);
}

impl<'a> AsyncVectorExt for RawDeviceBufferRef<'a, f32> {
  type Ctx = DeviceCtxRef<'a>;
  type Vector = DeviceBufferRef<'a, f32>;

  fn async_vector_scale(&mut self, alpha: f32, ctx: &Self::Ctx) {
    if alpha == 1.0 {
      return;
    }
    let n = self.len();
    ctx.get_blas().set_pointer_mode(CublasPointerMode::Host);
    unsafe { cublas_sscal(
        &*ctx.get_blas(),
        n,
        alpha,
        self.as_mut_ptr(), 1,
    ) }.unwrap();
  }

  fn async_vector_add(&mut self, alpha: f32, x: &DeviceBufferRef<'a, f32>) {
    let n = self.len();
    let x_n = x.len();
    assert_eq!(n, x_n);

    x.ctx.get_blas().set_pointer_mode(CublasPointerMode::Host);
    unsafe { cublas_saxpy(
        &*x.ctx.get_blas(),
        n,
        alpha,
        x.as_ptr(), 1,
        self.as_mut_ptr(), 1,
    ) }.unwrap();
  }
}

pub trait BlasVectorExt {
  type Matrix;
  type Vector;

  fn row_vector_scale(&mut self, alpha: f32);
  fn row_vector_sum(&mut self, alpha: f32, x: &Self::Vector);
  fn matrix_vector_prod(&mut self,
      alpha: f32,
      a: &Self::Matrix, trans_a: Transpose,
      x: &Self::Vector,
  );
}

pub trait BlasMatrixExt {
  type Matrix;
  type Vector;

  fn matrix_scale(&mut self, alpha: f32);
  fn matrix_sum(&mut self, alpha: f32, x: &Self::Matrix);
  fn matrix_prod(&mut self,
      alpha: f32,
      a: &Self::Matrix, trans_a: Transpose,
      b: &Self::Matrix, trans_b: Transpose,
      beta: f32,
  );
}

pub trait AsyncBlasVectorExt<'ctx> {
  type Ctx;
  type Vector;

  fn async_vector_add(&self, alpha: f32, x: &Self::Vector, ctx: &'ctx Self::Ctx);
  fn async_vector_scale(&self, alpha: f32, ctx: &'ctx Self::Ctx);
}

impl<'ctx> AsyncBlasVectorExt<'ctx> for RawDeviceBufferRef<'ctx, f32> {
  type Ctx = DeviceCtxRef<'ctx>;
  type Vector = RawDeviceBufferRef<'ctx, f32>;

  fn async_vector_add(&self, alpha: f32, x: &RawDeviceBufferRef<'ctx, f32>, ctx: &'ctx DeviceCtxRef<'ctx>) {
    assert_eq!(self.len(), x.len());
    ctx.get_blas().set_pointer_mode(CublasPointerMode::Host);
    unsafe { cublas_saxpy(
        &*ctx.get_blas(),
        self.len(),
        alpha,
        x.as_ptr(), 1,
        self.as_mut_ptr(), 1,
    ) }.ok().expect("cublas saxpy failed");
  }

  fn async_vector_scale(&self, alpha: f32, ctx: &'ctx DeviceCtxRef<'ctx>) {
    ctx.get_blas().set_pointer_mode(CublasPointerMode::Host);
    unsafe { cublas_sscal(
        &*ctx.get_blas(),
        self.len(),
        alpha,
        self.as_mut_ptr(), 1,
    ) }.unwrap();
  }
}

impl<'a> BlasVectorExt for DeviceBufferRefMut<'a, f32> {
  type Matrix = DeviceBufferRef<'a, f32>;
  type Vector = DeviceBufferRef<'a, f32>;

  fn row_vector_scale(&mut self, alpha: f32) {
    let n = self.len();
    self.ctx.get_blas().set_pointer_mode(CublasPointerMode::Host);
    unsafe { cublas_sscal(
        &*self.ctx.get_blas(),
        n,
        alpha,
        self.as_mut_ptr(), 1,
    ) }.unwrap();
  }

  fn row_vector_sum(&mut self, alpha: f32, x: &DeviceBufferRef<f32>) {
    let n = self.len();
    let x_n = x.len();
    assert_eq!(n, x_n);

    self.ctx.get_blas().set_pointer_mode(CublasPointerMode::Host);
    unsafe { cublas_saxpy(
        &*self.ctx.get_blas(),
        n,
        alpha,
        x.as_ptr(), 1,
        self.as_mut_ptr(), 1,
    ) }.unwrap();
  }

  fn matrix_vector_prod(&mut self,
      alpha: f32,
      a: &Self::Matrix, trans_a: Transpose,
      x: &Self::Vector)
  {
    unimplemented!();
  }
}

impl<'a> BlasVectorExt for DeviceArray2dViewMut<'a, f32> {
  type Matrix = DeviceArray2dView<'a, f32>;
  type Vector = DeviceArray2dView<'a, f32>;

  fn row_vector_scale(&mut self, alpha: f32) {
    let (m, n) = self.bound();
    assert_eq!(m, 1);
    let incx = self.stride();
    self.data.ctx.get_blas().set_pointer_mode(CublasPointerMode::Host);
    unsafe { cublas_sscal(
        &*self.data.ctx.get_blas(),
        n,
        alpha,
        self.as_mut_ptr(), incx,
    ) }.unwrap();
  }

  fn row_vector_sum(&mut self, alpha: f32, x: &DeviceArray2dView<f32>) {
    let (m, n) = self.bound();
    let (x_m, x_n) = x.bound();
    assert_eq!(n, x_n);
    assert_eq!(m, 1);
    assert_eq!(x_m, 1);
    let incy = self.stride();
    let incx = x.stride();

    self.data.ctx.get_blas().set_pointer_mode(CublasPointerMode::Host);
    unsafe { cublas_saxpy(
        &*self.data.ctx.get_blas(),
        n,
        alpha,
        x.as_ptr(), incx,
        self.as_mut_ptr(), incy,
    ) }.unwrap();
  }

  fn matrix_vector_prod(&mut self,
      alpha: f32,
      a: &Self::Matrix, trans_a: Transpose,
      x: &Self::Vector)
  {
    unimplemented!();
  }
}

impl<'a> BlasMatrixExt for DeviceArray2dViewMut<'a, f32> {
  type Matrix = DeviceArray2dView<'a, f32>;
  type Vector = DeviceArray2dView<'a, f32>;

  fn matrix_scale(&mut self, alpha: f32) {
    if self.bound().0 == self.stride() {
      self.data.ctx.get_blas().set_pointer_mode(CublasPointerMode::Host);
      unsafe { cublas_sscal(
          &*self.data.ctx.get_blas(),
          self.bound().len(),
          alpha,
          self.as_mut_ptr(), 1,
      ) }.unwrap();
    } else {
      panic!("strided shapes not supported yet for .matrix_scale()!");
    }
  }

  fn matrix_sum(&mut self, alpha: f32, x: &DeviceArray2dView<f32>) {
    let (m, n) = self.bound();
    let (x_m, x_n) = x.bound();
    assert_eq!(n, x_n);
    assert_eq!(m, x_m);
    if self.bound().0 == self.stride() && x.bound().0 == x.stride() {
      self.data.ctx.get_blas().set_pointer_mode(CublasPointerMode::Host);
      unsafe { cublas_saxpy(
          &*self.data.ctx.get_blas(),
          m * n,
          alpha,
          x.as_ptr(), 1,
          self.as_mut_ptr(), 1,
      ) }.unwrap();
    } else {
      panic!("strided shapes not supported yet for .matrix_sum()!");
    }
  }

  fn matrix_prod(&mut self,
      alpha: f32,
      a: &DeviceArray2dView<f32>, trans_a: Transpose,
      b: &DeviceArray2dView<f32>, trans_b: Transpose,
      beta: f32)
  {
    let (m, n) = self.bound();
    let (a_m, a_n) = a.bound();
    let (b_m, b_n) = b.bound();
    let (at_m, at_n) = match trans_a {
      Transpose::N => (a_m, a_n),
      Transpose::T => (a_n, a_m),
    };
    let (bt_m, bt_n) = match trans_b {
      Transpose::N => (b_m, b_n),
      Transpose::T => (b_n, b_m),
    };
    assert_eq!(m, at_m);
    assert_eq!(n, bt_n);
    assert_eq!(at_n, bt_m);
    let k = at_n;
    let lda = a.stride();
    let ldb = b.stride();
    let ldc = self.stride();

    // FIXME(20160201): check same devices.
    self.data.ctx.get_blas().set_pointer_mode(CublasPointerMode::Host);
    unsafe { cublas_sgemm(
        &*self.data.ctx.get_blas(),
        trans_a.to_cublas(), trans_b.to_cublas(),
        m, n, k,
        alpha,
        a.as_ptr(), lda,
        b.as_ptr(), ldb,
        beta,
        self.as_mut_ptr(), ldc,
    ) }.unwrap();
  }
}
