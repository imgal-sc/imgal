use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::client::ComputeClient;

/// Initialize access to the GPU using Wgpu.
pub struct Gpu {
    device: WgpuDevice,
    client: ComputeClient<WgpuRuntime>,
}

impl Gpu {
    pub fn init() -> Self{
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        Self {
          device,
          client
        }
    }
}
