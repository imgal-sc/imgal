use ndarray::{ArrayBase, ArrayD, AsArray, Axis, Dimension, ViewRepr, Zip};

use crate::prelude::*;
use crate::statistics::sum;

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
pub fn sum_project<'a, T, A, D>(data: A, axis: Option<usize>) -> Result<ArrayD<T>, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let mut shape = data.shape().to_vec();
    let axis = axis.unwrap_or(shape.len().saturating_sub(1));
    shape.remove(axis);
    let mut res = ArrayD::from_elem(shape, T::default());
    let data = data.into_dyn();
    let lanes = data.lanes(Axis(axis));
    Zip::from(lanes).and(res.view_mut()).for_each(|ln, v| {
        *v = sum(ln, None);
    });
    Ok(res)
}
