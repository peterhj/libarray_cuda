pub use device::array::{DeviceArray2d, DeviceArray3d};
pub use device::context::{DeviceContext, DeviceCtxRef};
pub use device::memory::{DeviceBuffer};

pub mod array;
pub mod comm;
pub mod context;
pub mod ext;
pub mod linalg;
pub mod memory;
pub mod sync;
