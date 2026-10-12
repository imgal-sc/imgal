use core::mem::size_of;
use std::marker::PhantomData;
use std::sync::OnceLock;

use cubecl::bytes::Bytes;
use cubecl::server::Handle;
use cubecl::std::tensor::compact_strides;
use cubecl::{Device, prelude::*};
use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr};

pub(crate) static GPU_CLIENT: OnceLock<Client> = OnceLock::new();
pub(crate) static GPU_DEVICE: OnceLock<Device> = OnceLock::new();
pub(crate) const GPU_CLIENT_FAIL_MSG: &str = "Failed to obtain the GPU client.";
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
        let handle: Handle;
        let shape = data.shape();
        let size = shape.iter().product::<usize>() * size_of::<F>();
        let create_handle = |s: &[F]| {
            let buf = client.empty(size);
            let bytes = Bytes::from_elems(F::as_bytes(s).to_vec());
            client.write(&buf, bytes);
            buf
        };
        if let Some(s) = data.as_slice_memory_order() {
            handle = create_handle(s);
        } else {
            let mut buf: Vec<F> = Vec::with_capacity(shape.iter().product::<usize>());
            data.rows().into_iter().for_each(|r| {
                if let Some(s) = r.as_slice_memory_order() {
                    buf.extend_from_slice(s);
                } else {
                    buf.extend_from_slice(&r.to_vec());
                }
            });
            handle = create_handle(&buf);
        }
        Self {
            data: handle,
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
    let device = Device::wgpu(Default::default()).expect(GPU_INIT_FAIL_MSG);
    let client = device.client();
    GPU_DEVICE.get_or_init(|| device);
    GPU_CLIENT.get_or_init(|| client);
}
