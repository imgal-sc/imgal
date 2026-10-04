use core::mem::size_of;

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr};

use crate::gpu::Gpu;

pub fn fftf<'a, A, D>(data: A) -> ndarray::Array<f32, D>
where
    A: AsArray<'a, f32, D>,
    D: Dimension,
{
    let data: ArrayBase<ViewRepr<&'a f32>, D> = data.into();
    let size = data.len();
    let gpu = Gpu::init();
    let (raw, _) = data.to_owned().into_raw_vec_and_offset();
    let input_handle = gpu.client.create_from_slice(f32::as_bytes(&raw));
    let output_handle = gpu.client.empty(size_of::<f32>() * size);
    // 256 is a good starting point but perhaps this should be configurable?
    let cube_dim = CubeDim::new_1d(512);
    let cube_count = calculate_cube_count_elemwise(&gpu.client, size, cube_dim);
    dbg!(gpu.client.memory_usage().unwrap());
    unsafe {
        gpu_fftf::launch::<WgpuRuntime>(
            &gpu.client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(input_handle, size),
            ArrayArg::from_raw_parts(output_handle.clone(), size),
            size,
        );
    }
    let raw_output = gpu.client.read_one_unchecked(output_handle);
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
