use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

/// Initialize access to the GPU using Wgpu.
pub struct Gpu {
    pub device: WgpuDevice,
    pub client: ComputeClient<WgpuRuntime>,
}

impl Gpu {
    pub fn init() -> Self {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        Self { device, client }
    }
}
