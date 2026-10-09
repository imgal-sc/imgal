use core::mem::size_of;
use std::marker::PhantomData;
use std::sync::OnceLock;

use cubecl::bytes::Bytes;
use cubecl::server::Handle;
use cubecl::std::tensor::{TensorHandle, compact_strides};
use cubecl::{Device, prelude::*};
use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr};

pub(crate) static GPU_DEVICE: OnceLock<Device> = OnceLock::new();
pub(crate) const GPU_DEVICE_FAIL_MSG: &str = "Failed to obtain the GPU device.";
pub(crate) const GPU_INIT_FAIL_MSG: &str = "Failed to initialize the GPU.";

/// TODO
#[derive(Debug, Clone)]
pub struct GpuTensor<F>
where
    F: Float + CubeElement,
{
    data: Handle,
    shape: Vec<usize>,
    _f: PhantomData<F>,
}

impl<F> GpuTensor<F>
where
    F: Float + CubeElement,
{
    /// TODO
    pub fn as_arg(&self) -> TensorArg {
        unsafe {
            TensorArg::from_raw_parts(
                self.data.clone(),
                compact_strides(&self.shape),
                self.shape.clone().into(),
            )
        }
    }

    /// TODO
    pub fn empty(shape: Vec<usize>, client: &Client) -> GpuTensor<F> {
        let size = shape.iter().product::<usize>() * size_of::<F>();
        let data = client.empty(size);
        Self {
            data,
            shape,
            _f: PhantomData,
        }
    }

    /// TODO
    pub fn new<'a, A, D>(data: A, client: &Client) -> GpuTensor<F>
    where
        A: AsArray<'a, F, D>,
        D: Dimension,
    {
        let data: ArrayBase<ViewRepr<&'a F>, D> = data.into();
        let th: TensorHandle;
        let shape = data.shape();
        if let Some(s) = data.as_slice_memory_order() {
            th = TensorHandle::new_contiguous(
                shape,
                client.create_from_slice(F::as_bytes(s)),
                F::elem_type_native(),
            );
        } else {
            let size = shape.iter().product::<usize>() * size_of::<F>();
            let mut buf: Vec<u8> = Vec::with_capacity(size);
            data.rows().into_iter().for_each(|r| {
                if let Some(s) = r.as_slice_memory_order() {
                    buf.extend_from_slice(F::as_bytes(s));
                } else {
                    buf.extend_from_slice(F::as_bytes(&r.to_vec()));
                }
            });
            th = TensorHandle::new_contiguous(
                shape,
                client.create(Bytes::from_bytes_vec(buf)),
                F::elem_type_native(),
            );
        }
        Self {
            data: th.handle,
            shape: shape.to_vec(),
            _f: PhantomData,
        }
    }

    /// TODO
    pub fn read(self, client: &Client) -> Vec<F> {
        let bytes = client
            .read_one(self.data)
            .expect("Failed to read data from the Tensor.");
        F::from_bytes(&bytes).to_vec()
    }
}
/// Initialize the GPU and store the GPU device.
#[inline(always)]
pub fn init_gpu() {
    GPU_DEVICE.get_or_init(|| Device::wgpu(Default::default()).expect(GPU_INIT_FAIL_MSG));
}
