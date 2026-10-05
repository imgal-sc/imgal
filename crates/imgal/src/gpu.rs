use std::sync::OnceLock;

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use ndarray::{ArrayView, Dimension};

pub(crate) type GpuRuntime = WgpuRuntime;
pub(crate) static GPU_CLIENT: OnceLock<ComputeClient<WgpuRuntime>> = OnceLock::new();

/// TODO
/// This triggers JIT shader compilation on the host.
#[inline(always)]
pub fn init_gpu() {
    GPU_CLIENT.get_or_init(|| {
        let device = WgpuDevice::default();
        WgpuRuntime::client(&device)
    });
}

/// TODO
/// This converts an n-dimensional array into a Handle that can be loaded onto
/// GPU.
#[inline(always)]
pub fn to_handle<D>(data: ArrayView<f32, D>, client: &ComputeClient<GpuRuntime>) -> Handle
where
    D: Dimension,
{
    let handle: Handle;
    if let Some(s) = data.as_slice_memory_order() {
        handle = client.create_from_slice(f32::as_bytes(s));
    } else {
        let mut buf: Vec<u8> = Vec::with_capacity(size_of::<f32>() * data.len());
        data.rows().into_iter().for_each(|r| {
            if let Some(s) = r.as_slice_memory_order() {
                buf.extend_from_slice(f32::as_bytes(s));
            } else {
                buf.extend(f32::as_bytes(&r.to_vec()));
            }
        });
        handle = client.create(Bytes::from_bytes_vec(buf));
    }
    handle
}
