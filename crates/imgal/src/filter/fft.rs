use cubecl::prelude::*;
use ndarray::{AsArray, ArrayBase, Dimension, ViewRepr};

use crate::prelude::*;
use crate::gpu::Gpu;

pub fn fftf<'a, T, A, D>(data: A)
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let size = data.len();
    let gpu = Gpu::init();
    todo!();
}

/// This protoype just multiplies the input values of an array by 0.5. 
#[cube(launch)]
fn gpu_fftf(input: &Array<f32>, output: &mut Array<f32>, #[comptime] size: usize){
    let idx = ABSOLUTE_POS as usize;
    if idx < size {
        output[idx] = 0.5 * input[idx];
    }
}
