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
pub struct GpuTensor<N>
where
    N: Numeric + CubeElement,
{
    /// The raw GPU contiguous value buffer.
    pub data: Handle,
    /// TODO
    pub shape: Vec<usize>,
    _n: PhantomData<N>,
}

impl<N> GpuTensor<N>
where
    N: Numeric + CubeElement,
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

    /// Initialize an empty GPU tensor of the given size.
    pub fn empty(shape: Vec<usize>) -> GpuTensor<N> {
        let client = GPU_CLIENT.get().expect(GPU_CLIENT_FAIL_MSG);
        let size = shape.iter().product::<usize>() * size_of::<N>();
        Self {
            data: client.empty(size),
            shape,
            _n: PhantomData,
        }
    }

    /// Initialize a GPU tensor of size 1.
    ///
    /// This is useful for functions that reduce an input tensor down to one
    /// value.
    pub fn empty_one() -> GpuTensor<N> {
        let client = GPU_CLIENT.get().expect(GPU_CLIENT_FAIL_MSG);
        Self {
            data: client.empty(1 * size_of::<N>()),
            shape: vec![1],
            _n: PhantomData,
        }
    }

    /// TODO
    pub fn new<'a, A, D>(data: A) -> GpuTensor<N>
    where
        A: AsArray<'a, N, D>,
        D: Dimension,
    {
        let client = GPU_CLIENT.get().expect(GPU_CLIENT_FAIL_MSG);
        let data: ArrayBase<ViewRepr<&'a N>, D> = data.into();
        let handle: Handle;
        let shape = data.shape();
        let size = shape.iter().product::<usize>() * size_of::<N>();
        let create_handle = |d: Vec<N>| {
            let buf = client.empty(size);
            let bytes = Bytes::from_elems(d);
            client.write(&buf, bytes);
            buf
        };
        if let Some(s) = data.as_slice_memory_order() {
            handle = create_handle(s.to_vec());
        } else {
            let mut buf: Vec<N> = Vec::with_capacity(shape.iter().product::<usize>());
            data.rows().into_iter().for_each(|r| {
                if let Some(s) = r.as_slice_memory_order() {
                    buf.extend_from_slice(s);
                } else {
                    buf.extend_from_slice(&r.to_vec());
                }
            });
            handle = create_handle(buf);
        }
        Self {
            data: handle,
            shape: shape.to_vec(),
            _n: PhantomData,
        }
    }

    /// TODO
    pub fn read(self) -> ArrayD<N> {
        let client = GPU_CLIENT.get().expect(GPU_CLIENT_FAIL_MSG);
        let bytes = client
            .read_one(self.data)
            .expect("Failed to read data from the Tensor.");
        Array::from_shape_vec(self.shape, N::from_bytes(&bytes).to_vec()).unwrap()
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
