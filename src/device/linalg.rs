use device::array::{DeviceArray2dView, DeviceArray2dViewMut};
use device::context::{DeviceCtxRef};
use device::memory::{RawDeviceBuffer};

use array_new::{Shape, ArrayView, ArrayViewMut};
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
}

impl<'ctx> AsyncBlasVectorExt<'ctx> for RawDeviceBuffer<f32> {
  type Ctx = DeviceCtxRef<'ctx>;
  type Vector = RawDeviceBuffer<f32>;

  fn async_vector_add(&self, alpha: f32, x: &RawDeviceBuffer<f32>, ctx: &'ctx DeviceCtxRef<'ctx>) {
    assert_eq!(self.len(), x.len());
    ctx.blas.set_pointer_mode(CublasPointerMode::Host);
    unsafe { cublas_saxpy(
        &ctx.blas,
        self.len(),
        alpha,
        x.as_ptr(), 1,
        self.as_mut_ptr(), 1,
    ) }.ok().expect("cublas saxpy failed");
  }
}

impl<'a> BlasVectorExt for DeviceArray2dViewMut<'a, f32> {
  type Matrix = DeviceArray2dView<'a, f32>;
  type Vector = DeviceArray2dView<'a, f32>;

  fn row_vector_scale(&mut self, alpha: f32) {
    let (m, n) = self.bound();
    assert_eq!(m, 1);
    let incx = self.stride();
    self.data.ctx.blas.set_pointer_mode(CublasPointerMode::Host);
    unsafe { cublas_sscal(
        &self.data.ctx.blas,
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

    self.data.ctx.blas.set_pointer_mode(CublasPointerMode::Host);
    unsafe { cublas_saxpy(
        &self.data.ctx.blas,
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
      self.data.ctx.blas.set_pointer_mode(CublasPointerMode::Host);
      unsafe { cublas_sscal(
          &self.data.ctx.blas,
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
      self.data.ctx.blas.set_pointer_mode(CublasPointerMode::Host);
      unsafe { cublas_saxpy(
          &self.data.ctx.blas,
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

    self.data.ctx.blas.set_pointer_mode(CublasPointerMode::Host);
    unsafe { cublas_sgemm(
        &self.data.ctx.blas,
        trans_a.to_cublas(), trans_b.to_cublas(),
        m, n, k,
        alpha,
        a.as_ptr(), lda,
        b.as_ptr(), ldb,
        beta,
        self.as_mut_ptr(), ldc,
    ) }.unwrap();
    unimplemented!();
  }
}
