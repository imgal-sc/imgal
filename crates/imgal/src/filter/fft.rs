use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr};

use crate::gpu::*;

/// [WIP] GPU accelerated Fast Fourier Transforms (FFT)
pub fn fft<'a, A, D>(data: A) -> ndarray::Array<f32, D>
where
    A: AsArray<'a, f32, D>,
    D: Dimension,
{
    let data: ArrayBase<ViewRepr<&'a f32>, D> = data.into();
    let shape = data.raw_dim();
    let size = data.len();
    init_gpu();
    let client = GPU_DEVICE.get().expect(GPU_DEVICE_FAIL_MSG).client();
    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise(&client, size, cube_dim);
    let input = GpuTensor::new(&data, &client);
    let output = GpuTensor::<f32>::empty(data.shape().to_vec(), &client);
    unsafe {
        gk_fft::launch_unchecked::<f32>(
            &client,
            cube_count,
            cube_dim,
            input.as_arg(),
            output.as_arg(),
            size,
        );
    }
    ndarray::Array::from_shape_vec(shape, output.read(&client)).unwrap()
}

/// [WIP] Prototype FFT GPU kernel. For now this performs a simple value halving
/// instead of a performing an actual FFT.
#[cube(launch_unchecked)]
fn gk_fft<F: Float>(input: &Tensor<f32>, output: &mut Tensor<f32>, size: usize) {
    let idx = ABSOLUTE_POS as usize;
    if idx < size {
        output[idx] = 0.5 * input[idx];
    }
}
