use device::array::{DeviceArray2dView, DeviceArray2dViewMut};
use device::context::{DeviceCtxRef};
use device::memory::{DeviceBufferRef, DeviceBufferRefMut, RawDeviceBufferRef};

use array_new::{Shape, ArrayView, ArrayViewMut};

pub trait SamplingDistribution {
}

pub struct UniformDist;

impl SamplingDistribution for UniformDist {}

pub struct GaussianDist<T> {
  pub mean: T,
  pub std:  T,
}

impl<T> SamplingDistribution for GaussianDist<T> {}

pub trait RandomSampleExt<Dist> where Dist: SamplingDistribution {
  fn sample(&mut self, dist: &Dist);
}

impl<'a> RandomSampleExt<UniformDist> for DeviceBufferRefMut<'a, f32> {
  fn sample(&mut self, _dist: &UniformDist) {
    match unsafe { self.ctx.get_rng().generate_uniform(self.as_mut_ptr(), self.len()) } {
      Ok(_) => {}
      Err(e) => panic!("sample: failed to generate uniform"),
    }
  }
}

impl<'a> RandomSampleExt<GaussianDist<f32>> for DeviceBufferRefMut<'a, f32> {
  fn sample(&mut self, dist: &GaussianDist<f32>) {
    match unsafe { self.ctx.get_rng().generate_normal(self.as_mut_ptr(), self.len(), dist.mean, dist.std) } {
      Ok(_) => {}
      Err(e) => panic!("sample: failed to generate gaussian"),
    }
  }
}
