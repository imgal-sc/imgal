use core::mem::size_of;

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr};

use crate::gpu::{GPU_CLIENT, warm_gpu};

pub fn fftf<'a, A, D>(data: A) -> ndarray::Array<f32, D>
where
    A: AsArray<'a, f32, D>,
    D: Dimension,
{
    let data: ArrayBase<ViewRepr<&'a f32>, D> = data.into();
    let size = data.len();
    warm_gpu();
    let (raw, _) = data.to_owned().into_raw_vec_and_offset();
    let client = GPU_CLIENT.get().expect("Failed to initialize the GPU.");
    let input_handle = client.create_from_slice(f32::as_bytes(&raw));
    let output_handle = client.empty(size_of::<f32>() * size);
    // 256 is a good starting point but perhaps this should be configurable?
    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise(client, size, cube_dim);
    unsafe {
        gpu_fftf::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(input_handle, size),
            ArrayArg::from_raw_parts(output_handle.clone(), size),
            size,
        );
    }
    let raw_output = client.read_one_unchecked(output_handle);
    let res = f32::from_bytes(&raw_output).to_vec();
    ndarray::Array::from_shape_vec(data.raw_dim(), res).unwrap()
}

/// This protoype just multiplies the input values of an array by 0.5.
#[cube(launch)]
fn gpu_fftf(input: &Array<f32>, output: &mut Array<f32>, #[comptime] size: usize) {
    let idx = ABSOLUTE_POS as usize;
    if idx < size {
        output[idx] = 0.5 * input[idx];
    }
}
