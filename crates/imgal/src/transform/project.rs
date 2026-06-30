use ndarray::{Array, ArrayBase, AsArray, Axis, Dimension, RemoveAxis, ViewRepr, Zip};
use rustfft::num_traits::Zero;

use crate::prelude::*;

/// Project an n-dimensional image by summing along a specified axis.
///
/// # Description
///
/// Computes the sum projection of an n-dimensional image along the specified
/// axis. Each output element is the sum of all values along the corresponding
/// lane of the projection axis. The resulting image has one fewer dimension
/// than the input image.
///
/// # Arguments
///
/// * `data`: The input n-dimensional image.
/// * `axis`: The axis to sum project along. If `None` then the last axis is
///   used.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `Ok(Array<T, D::Smaller>)`: The sum projected image.
/// * `Err(ImgalError)`: If `axis` is greater than or equal to the number of
///   dimensions.
pub fn sum_project<'a, T, A, D>(
    data: A,
    axis: Option<usize>,
    threads: Option<usize>,
) -> Result<Array<T, D::Smaller>, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension + RemoveAxis,
    T: 'a + AsNumeric + Zero,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let n_dims = data.ndim();
    let axis = axis.unwrap_or(n_dims.saturating_sub(1));
    if axis > n_dims {
        return Err(ImgalError::InvalidParameterValueGreater {
            param_name: "axis",
            value: n_dims,
        });
    }
    let lanes = data.lanes(Axis(axis));
    Ok(par!(threads,
        seq_exp: Zip::from(lanes).map_collect(|l| l.sum()),
        par_exp: Zip::from(lanes).par_map_collect(|l| l.sum())))
}
