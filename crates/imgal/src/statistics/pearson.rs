use ndarray::{ArrayBase, AsArray, Ix1, ViewRepr, Zip};

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Compute the Pearson correlation coefficient between two 1D arrays.
///
/// # Description
///
/// Computes the Pearson correlation coefficient, a measure of linear
/// correlation between two sets of 1D data.
///
/// Pearson's correlation coefficient is computed as:
/// ```text
/// r = Σ[(aᵢ - mean(a)) × (bᵢ - mean(b))] / √[Σ(aᵢ - mean(a))² × Σ(bᵢ - mean(b))²]
/// ```
///
/// # Arguments
///
/// * `data_a`: The first array for correlation analysis.
/// * `data_b`: The second array for correlation analysis.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Ok(f64)`: Pearson's correlatoin coefficient ranging between `-1.0`
///   (perfect negative correlation), `0.0` (no correlation), and `1.0`
///   (perfect positive correlation).
/// * `Err(ImgalError)`: If `data_a.len() != data_b.len()`
pub fn pearson_correlation<'a, T, A>(
    data_a: A,
    data_b: A,
    parallel: bool,
) -> Result<f64, ImgalError>
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
    let n = n as f64;
    let (sum_a, sum_b) = if parallel {
        Zip::from(data_a).and(data_b).par_fold(
            || (0.0, 0.0),
            |acc, &a, &b| (acc.0 + a.to_f64(), acc.1 + b.to_f64()),
            |acc, res| (acc.0 + res.0, acc.1 + res.1),
        )
    } else {
        Zip::from(data_a)
            .and(data_b)
            .fold((0.0, 0.0), |acc, &a, &b| {
                (acc.0 + a.to_f64(), acc.1 + b.to_f64())
            })
    };
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;
    let (numer, sq_a, sq_b) = if parallel {
        Zip::from(data_a).and(data_b).par_fold(
            || (0.0, 0.0, 0.0),
            |acc, &a, &b| {
                let diff_a = a.to_f64() - mean_a;
                let diff_b = b.to_f64() - mean_b;
                (
                    acc.0 + diff_a * diff_b,
                    acc.1 + diff_a * diff_a,
                    acc.2 + diff_b * diff_b,
                )
            },
            |acc, res| (acc.0 + res.0, acc.1 + res.1, acc.2 + res.2),
        )
    } else {
        Zip::from(data_a)
            .and(data_b)
            .fold((0.0, 0.0, 0.0), |acc, &a, &b| {
                let diff_a = a.to_f64() - mean_a;
                let diff_b = b.to_f64() - mean_b;
                (
                    acc.0 + diff_a * diff_b,
                    acc.1 + diff_a * diff_a,
                    acc.2 + diff_b * diff_b,
                )
            })
    };
    let denominator = (sq_a * sq_b).sqrt();
    if denominator == 0.0 {
        return Err(ImgalError::InvalidGeneric {
            msg: "Cannot compute Pearson correlation. One or both arrays have zero variance.",
        });
    }
    Ok(numer / denominator)
}
