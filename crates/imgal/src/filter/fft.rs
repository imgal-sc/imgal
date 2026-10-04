use core::mem::size_of;

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::server::Handle;
use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr};

use crate::gpu::{GPU_CLIENT, GpuRuntime, warm_gpu};

pub fn fft<'a, A, D>(data: A) -> ndarray::Array<f32, D>
where
    A: AsArray<'a, f32, D>,
    D: Dimension,
{
    let data: ArrayBase<ViewRepr<&'a f32>, D> = data.into();
    let shape = data.raw_dim();
    let size = data.len();
    warm_gpu();
    let client = GPU_CLIENT.get().expect("Failed to initialize the GPU.");
    let cube_dim = CubeDim::new_1d(256);
    // 256 is a good starting point but perhaps this should be configurable?
    let cube_count = calculate_cube_count_elemwise(client, size, cube_dim);
    let out_handle = client.empty(size_of::<f32>() * size);
    let in_handle = slice_to_handle(data.as_slice_memory_order().unwrap(), client);
    // let in_handle: Handle;
    // if let Some(s) = data.as_slice_memory_order() {
    //     in_handle = slice_to_handle(s, client);
    // } else {
    //     data.rows().into_iter().for_each(|r| {
    //         if let Some(s) = r.as_slice_memory_order() {
    //             todo!("Implement per slice handle creation.");
    //         } else {
    //             todo!("Implement non-contiguious memory for handle creation.");
    //         }
    //     })
    // }
    unsafe {
        gpu_fft::launch::<GpuRuntime>(
            client,
            cube_count.clone(),
            cube_dim,
            ArrayArg::from_raw_parts(in_handle, size),
            ArrayArg::from_raw_parts(out_handle.clone(), size),
            size,
        )
    }
    let raw_output = client.read_one_unchecked(out_handle);
    let res = f32::from_bytes(&raw_output).to_vec();
    ndarray::Array::from_shape_vec(shape, res).unwrap()
}

/// This protoype just multiplies the input values of an array by 0.5.
#[cube(launch)]
fn gpu_fft(input: &Array<f32>, output: &mut Array<f32>, #[comptime] size: usize) {
    let idx = ABSOLUTE_POS as usize;
    if idx < size {
        output[idx] = 0.5 * input[idx];
    }
}

#[inline(always)]
fn slice_to_handle(x: &[f32], client: &ComputeClient<GpuRuntime>) -> Handle {
    client.create_from_slice(f32::as_bytes(x))
}
