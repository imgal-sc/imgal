use core::mem::size_of;
use std::marker::PhantomData;
use std::sync::OnceLock;

use cubecl::bytes::Bytes;
use cubecl::server::Handle;
use cubecl::std::tensor::TensorHandle;
use cubecl::{Device, prelude::*};
use ndarray::{ArrayView, Dimension};

use crate::prelude::*;

pub(crate) static GPU_DEVICE: OnceLock<Device> = OnceLock::new();
pub(crate) const GPU_INIT_FAIL_MSG: &str = "Failed to initialize the GPU.";

#[derive(Debug, Clone)]
pub struct GpuTensor<T: AsNumeric + CubeElement> {
    data: Handle,
    shape: Vec<usize>,
    _t: PhantomData<T>,
}

impl<T: AsNumeric + CubeElement> GpuTensor<T> {
    pub fn empty(shape: Vec<usize>, client: &Client) {
        let size = shape.iter().product::<usize>() * size_of::<T>();
        todo!();
    }
}
// /// Initialize the GPU and store the GPU device.
// #[inline(always)]
// pub fn init_gpu() {
//     GPU_DEVICE.get_or_init(|| Device::wgpu(Default::default()).expect(GPU_INIT_FAIL_MSG));
// }

// /// Reserve `size` bytes of memory on the GPU.
// #[inline(always)]
// pub(crate) fn reserve_gpu_mem(size: usize, device: &Device) -> Handle {
//     let client = device.client();
//     client.empty(size_of::<f32>() * size)
// }

// /// Get raw data from the GPU.
// #[inline(always)]
// pub(crate) fn from_gpu(handle: Handle, device: &Device) -> Vec<f32> {
//     let client = device.client();
//     let raw = client.read_one_unchecked(handle);
//     f32::from_bytes(&raw).to_vec()
// }

// /// Send an n-dimensional ArrayView to the GPU.
// #[inline(always)]
// pub(crate) fn to_gpu<D>(
//     view: ArrayView<f32, D>,
//     client: &ComputeClient<GpuRuntime>,
// ) -> TensorHandle<GpuRuntime>
// where
//     D: Dimension,
// {
//     let th: TensorHandle<GpuRuntime>;
//     let shape = view.shape();
//     if let Some(s) = view.as_slice_memory_order() {
//         th = TensorHandle::new_contiguous(
//             shape,
//             client.create_from_slice(f32::as_bytes(s)),
//             f32::as_type_native_unchecked().storage_type(),
//         );
//     } else {
//         let mut buf: Vec<u8> = Vec::with_capacity(size_of::<f32>() * view.len());
//         view.rows().into_iter().for_each(|r| {
//             if let Some(s) = r.as_slice_memory_order() {
//                 buf.extend_from_slice(f32::as_bytes(s));
//             } else {
//                 buf.extend(f32::as_bytes(&r.to_vec()));
//             }
//         });
//         th = TensorHandle::new_contiguous(
//             shape,
//             client.create(Bytes::from_bytes_vec(buf)),
//             f32::as_type_native_unchecked().storage_type(),
//         );
//     }
//     th
// }
