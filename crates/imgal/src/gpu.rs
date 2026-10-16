//! GPU device setup and tensors.
//!
//! This module provides the functions and structs needed to initialize a new
//! GPU device, create/read/write tensors, and convert them into GPU kernel
//! arguments.

use core::mem::size_of;
use std::marker::PhantomData;
use std::sync::OnceLock;

use cubecl::bytes::Bytes;
use cubecl::server::Handle;
use cubecl::std::tensor::compact_strides;
use cubecl::{Device, prelude::*};
use ndarray::{ArrayBase, ArrayD, AsArray, Dimension, ViewRepr};

pub(crate) static GPU_CLIENT: OnceLock<Client> = OnceLock::new();
pub(crate) const GPU_CLIENT_FAIL_MSG: &str = "Failed to obtain the GPU client.";
pub(crate) const GPU_INIT_FAIL_MSG: &str = "Failed to initialize the GPU.";

/// A GPU tensor.
///
/// The `GpuTensor` represents array data (*i.e.* a tensor) on an initialized
/// GPU. An active `GpuTensor` will always have *some* GPU memory allocation
/// and a shape.
#[derive(Debug, Clone)]
pub struct GpuTensor<N>
where
    N: Numeric + CubeElement,
{
    /// The raw GPU contiguous value buffer.
    data: Handle,
    /// The shape of the tensor.
    pub shape: Vec<usize>,
    _n: PhantomData<N>,
}

impl<N> GpuTensor<N>
where
    N: Numeric + CubeElement,
{
    /// Convert this tensor into an argument for GPU kernels.
    ///
    /// # Description
    ///
    /// Converts this tensor into an argument that can be passed to a GPU kernel.
    ///
    /// # Returns
    ///
    /// * `TensorArg`: The tensor argument.
    pub fn as_arg(&self) -> TensorArg {
        unsafe {
            TensorArg::from_raw_parts(
                self.data.clone(),
                compact_strides(&self.shape),
                self.shape.clone().into(),
            )
        }
    }

    /// Create a new empty GPU tensor.
    ///
    /// # Description
    ///
    /// Creates a new empty GPU tensor with a size defined by `shape`.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the empty tensor.
    ///
    /// # Returns
    ///
    /// * `GpuTensor<N>`: An initialized empty GPU tensor.
    pub fn empty(shape: &[usize]) -> GpuTensor<N> {
        let client = GPU_CLIENT.get().expect(GPU_CLIENT_FAIL_MSG);
        let size = shape.iter().product::<usize>() * size_of::<N>();
        Self {
            data: client.empty(size),
            shape: shape.to_vec(),
            _n: PhantomData,
        }
    }

    /// Create a new empty GPU tensor of size one.
    ///
    /// # Description
    ///
    /// Creates a new empty GPU tensor of size one. A size one tensor is useful
    /// for functions that reduce an input tensor into one value.
    ///
    /// # Returns
    ///
    /// * `GpuTensor<N>`: An initialized empty GPU tensor with a size of one.
    pub fn empty_one() -> GpuTensor<N> {
        let client = GPU_CLIENT.get().expect(GPU_CLIENT_FAIL_MSG);
        Self {
            data: client.empty(1 * size_of::<N>()),
            shape: vec![1],
            _n: PhantomData,
        }
    }

    /// Create a new GPU tensor with input data.
    ///
    /// # Description
    ///
    /// Creates a new GPU tensor from the input *n*-dimensional image.
    ///
    /// # Arguments
    ///
    /// * `data`: The input *n*-dimensional image.
    ///
    /// # Returns
    ///
    /// * `GpuTensor<N>`: An initialized GPU tensor with the input data.
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

    /// Read data from the GPU tensor.
    ///
    /// # Description
    ///
    /// Reads *n*-dimensional data from the tensor's reserved GPU memory.
    ///
    /// # Returns
    ///
    /// * `ArrayD<N>`: The tensor data.
    pub fn read(self) -> ArrayD<N> {
        let client = GPU_CLIENT.get().expect(GPU_CLIENT_FAIL_MSG);
        let bytes = client
            .read_one(self.data)
            .expect("Failed to read data from the Tensor.");
        ArrayD::from_shape_vec(self.shape, N::from_bytes(&bytes).to_vec()).unwrap()
    }

    /// Write data into the GPU tensor.
    ///
    /// # Description
    ///
    /// Writes *n*-dimensional data into the tensor's reserved GPU memory.
    ///
    /// # Arguments
    ///
    /// * `data`: The input *n*-dimensional image.
    pub fn write<'a, A, D>(self, data: A)
    where
        A: AsArray<'a, N, D>,
        D: Dimension,
    {
        let client = GPU_CLIENT.get().expect(GPU_CLIENT_FAIL_MSG);
        let data: ArrayBase<ViewRepr<&'a N>, D> = data.into();
        let bytes: Bytes;
        if let Some(s) = data.as_slice_memory_order() {
            bytes = Bytes::from_elems(s.to_vec());
        } else {
            let mut buf: Vec<N> = Vec::with_capacity(self.shape.iter().product::<usize>());
            data.rows().into_iter().for_each(|r| {
                if let Some(s) = r.as_slice_memory_order() {
                    buf.extend_from_slice(s);
                } else {
                    buf.extend_from_slice(&r.to_vec());
                }
            });
            bytes = Bytes::from_elems(buf);
        }
        client.write(&self.data, bytes);
    }
}
/// Initialize the GPU and cache the compute client.
///
/// # Description
///
/// Initializes the GPU and caches the compute client in a OnceLock. The client
/// enables data transfer to and from the GPU.
#[inline(always)]
pub fn init_gpu() {
    let device = Device::wgpu(Default::default()).expect(GPU_INIT_FAIL_MSG);
    GPU_CLIENT.get_or_init(|| device.client());
}
