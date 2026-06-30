use ndarray::{Array, ArrayBase, AsArray, Axis, Dimension, RemoveAxis, ViewRepr, Zip};
use rustfft::num_traits::Zero;

use crate::prelude::*;

/// TODO
///
/// # Description
///
/// todo
///
/// # Arguments
///
/// * `axis`: Default is the last axis.
///
/// # Returns
///
/// todo
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
