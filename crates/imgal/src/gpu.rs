use std::sync::OnceLock;

use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

pub static GPU_CLIENT: OnceLock<ComputeClient<WgpuRuntime>> = OnceLock::new();

/// TODO
#[inline(always)]
pub fn warm_gpu() {
    GPU_CLIENT.get_or_init(|| {
        let device = WgpuDevice::default();
        WgpuRuntime::client(&device)
    });
}
