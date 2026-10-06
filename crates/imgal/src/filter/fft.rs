use core::mem::size_of;

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr};

use crate::gpu::*;

pub fn fft<'a, A, D>(data: A) -> ndarray::Array<f32, D>
where
    A: AsArray<'a, f32, D>,
    D: Dimension,
{
    let data: ArrayBase<ViewRepr<&'a f32>, D> = data.into();
    let shape = data.raw_dim();
    let size = data.len();
    init_gpu();
    let client = GPU_CLIENT.get().expect("Failed to initialize the GPU.");
    let cube_dim = CubeDim::new_1d(256);
    // 256 is a good starting point but perhaps this should be configurable?
    let cube_count = calculate_cube_count_elemwise(client, size, cube_dim);
    let out_handle = client.empty(size_of::<f32>() * size);
    let in_handle = to_gpu(data, client);
    unsafe {
        gpu_fft::launch(
            client,
            cube_count.clone(),
            cube_dim,
            ArrayArg::from_raw_parts(in_handle.handle, size),
            ArrayArg::from_raw_parts(out_handle.clone(), size),
            size,
        )
    }
    ndarray::Array::from_shape_vec(shape, from_gpu(out_handle, &client)).unwrap()
}

/// This protoype just multiplies the input values of an array by 0.5.
#[cube(launch)]
fn gpu_fft(input: &Array<f32>, output: &mut Array<f32>, size: usize) {
    let idx = ABSOLUTE_POS as usize;
    if idx < size {
        output[idx] = 0.5 * input[idx];
    }
}
