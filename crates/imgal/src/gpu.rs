use std::sync::OnceLock;

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::std::tensor::TensorHandle;
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

/// Get raw data from the GPU.
#[inline(always)]
pub(crate) fn from_gpu(handle: Handle, client: &ComputeClient<GpuRuntime>) -> Vec<f32> {
    let raw = client.read_one_unchecked(handle);
    f32::from_bytes(&raw).to_vec()
}

/// Send an n-dimensional ArrayView to the GPU.
#[inline(always)]
pub(crate) fn to_gpu<D>(
    view: ArrayView<f32, D>,
    client: &ComputeClient<GpuRuntime>,
) -> TensorHandle<GpuRuntime>
where
    D: Dimension,
{
    let th: TensorHandle<GpuRuntime>;
    let shape = view.shape();
    if let Some(s) = view.as_slice_memory_order() {
        th = TensorHandle::new_contiguous(
            shape,
            client.create_from_slice(f32::as_bytes(s)),
            f32::as_type_native_unchecked().storage_type(),
        );
    } else {
        let mut buf: Vec<u8> = Vec::with_capacity(size_of::<f32>() * view.len());
        view.rows().into_iter().for_each(|r| {
            if let Some(s) = r.as_slice_memory_order() {
                buf.extend_from_slice(f32::as_bytes(s));
            } else {
                buf.extend(f32::as_bytes(&r.to_vec()));
            }
        });
        th = TensorHandle::new_contiguous(
            shape,
            client.create(Bytes::from_bytes_vec(buf)),
            f32::as_type_native_unchecked().storage_type(),
        );
    }
    th
}
