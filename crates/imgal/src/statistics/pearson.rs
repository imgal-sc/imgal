use ndarray::{ArrayBase, AsArray, Ix1, ViewRepr, Zip};

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// TODO
///
/// # Description
///
/// todo
///
/// # Arguments
///
/// * `data_a`:
/// * `data_b`:
///
/// # Returns
///
/// * `Ok(f64)`:
/// * `Err(ImgalError)`:
pub fn pearson_correlation<'a, T, A>(data_a: A, data_b: A) -> Result<f64, ImgalError>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let data_a: ArrayBase<ViewRepr<&'a T>, Ix1> = data_a.into();
    let data_b: ArrayBase<ViewRepr<&'a T>, Ix1> = data_b.into();
    let n = data_a.len();
    if n != data_b.len() {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "data_a",
            a_arr_len: n,
            b_arr_name: "data_b",
            b_arr_len: data_b.len(),
        });
    }
    let (sum_a, sum_b) = Zip::from(data_a)
        .and(data_b)
        .fold((0.0, 0.0), |acc, &a, &b| {
            (acc.0 + a.to_f64(), acc.1 + b.to_f64())
        });
    let mean_a = sum_a / n.to_f64();
    let mean_b = sum_b / n.to_f64();
    let (numer, sq_a, sq_b) = Zip::from(data_a)
        .and(data_b)
        .fold((0.0, 0.0, 0.0), |acc, &a, &b| {
            let diff_a = a.to_f64() - mean_a;
            let diff_b = b.to_f64() - mean_b;
            (
                acc.0 + diff_a * diff_b,
                acc.1 + diff_a * diff_a,
                acc.2 + diff_b * diff_b,
            )
        });
    Ok(numer / (sq_a * sq_b).sqrt())
}
