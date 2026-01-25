use ndarray::{ArrayBase, ArrayD, ArrayView1, AsArray, Dimension, Ix1, Ix2, ViewRepr, s};

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Create an n-dimensional meatballs image.
///
/// # Description
///
/// Use the "meatballs" algorithm.
///
/// # Arguments
///
/// * `centers`: Centers are in [p, D] shape.
/// * `radii`:
/// * `blob_values`:
/// * `background_value`:
/// * `shape`:
///
/// # Returns
///
/// * `Ok(Array<T, D>)`:
/// * `Err(ImgalError)`:
pub fn meatballs<'a, T, A, B, C>(
    centers: A,
    radii: B,
    intensities: C,
    background: T,
    shape: &[usize],
) -> Result<ArrayD<T>, ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    B: AsArray<'a, T, Ix1>,
    C: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let centers: ArrayBase<ViewRepr<&'a T>, Ix2> = centers.into();
    let radii: ArrayBase<ViewRepr<&'a T>, Ix1> = radii.into();
    let intensities: ArrayBase<ViewRepr<&'a T>, Ix1> = intensities.into();
    let n_blobs = centers.shape()[0];
    if n_blobs != radii.len() {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "centers",
            a_arr_len: n_blobs,
            b_arr_name: "radii",
            b_arr_len: radii.len(),
        });
    };
    if n_blobs != intensities.len() {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "centers",
            a_arr_len: n_blobs,
            b_arr_name: "radii",
            b_arr_len: intensities.len(),
        });
    }
    let mut blobs_arr = ArrayD::from_elem(shape, background);
    blobs_arr.view_mut().indexed_iter_mut().for_each(|(p, v)| {
        *v = T::from_f64((0..n_blobs).fold(0.0, |acc, i| {
            acc + meatball_contribution(
                p.as_array_view(),
                centers.slice(s![i, ..]),
                radii[i],
                intensities[i],
            )
        }));
    });

    Ok(blobs_arr)
}

/// TODO
fn meatball_contribution<T>(
    current_pos: ArrayView1<usize>,
    center_pos: ArrayView1<T>,
    radius: T,
    intensity: T,
) -> f64
where
    T: AsNumeric,
{
    let d_pos: Vec<f64> = current_pos
        .iter()
        .zip(center_pos.iter())
        .map(|(&cur, &cen)| cur as f64 - cen.to_f64())
        .collect();
    let dist_sq = d_pos.iter().fold(0.0, |acc, &v| acc + v * v);
    let sigma_sq = (radius * radius).to_f64();

    intensity.to_f64() * (-dist_sq / (2.0 * sigma_sq)).exp()
}
