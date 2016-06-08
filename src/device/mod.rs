//pub use device::array::{DeviceArray2d, DeviceArray3d};
pub use self::comm::{for_all_devices};
//pub use device::context::{DeviceContext, DeviceCtxRef};
//pub use device::memory::{DeviceBuffer};
pub use self::context::*;
pub use self::memory::*;

pub mod array;
pub mod comm;
pub mod context;
pub mod ext;
pub mod linalg;
pub mod memory;
pub mod random;
pub mod sync;
