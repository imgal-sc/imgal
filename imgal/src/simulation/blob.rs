use ndarray::{ArrayBase, ArrayD, ArrayView1, AsArray, Dimension, Ix1, Ix2, ViewRepr, s};

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Create an n-dimensional metaballs blob image.
///
/// # Description
///
/// Creates a simulated n-dimensional blobs image using a variant of Jim Blinn's
/// metaballs blob simulation algorithm. This function uses a Gaussian falloff
/// strategy to simulate a smooth and continuous edge with no sharp edges.
///
/// # Arguments
///
/// * `centers`: Centers are in [p, D] shape.
/// * `radii`:
/// * `intensities`:
/// * `falloffs`: The falloff value for the Gaussian falloff.....
/// * `background_value`:
/// * `shape`:
///
/// # Returns
///
/// * `Ok(ArrayD<f64>)`:
/// * `Err(ImgalError)`:
pub fn gaussian_metaballs<'a, T, A, B>(
    centers: A,
    radii: B,
    intensities: B,
    falloffs: B,
    background: T,
    shape: &[usize],
) -> Result<ArrayD<f64>, ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    B: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let centers: ArrayBase<ViewRepr<&'a T>, Ix2> = centers.into();
    let radii: ArrayBase<ViewRepr<&'a T>, Ix1> = radii.into();
    let intensities: ArrayBase<ViewRepr<&'a T>, Ix1> = intensities.into();
    let falloffs: ArrayBase<ViewRepr<&'a T>, Ix1> = falloffs.into();
    let background = background.to_f64();
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
    // TODO ensure shape length is the same as n_dims length
    let mut blobs_arr = ArrayD::from_elem(shape, background);
    blobs_arr.view_mut().indexed_iter_mut().for_each(|(p, v)| {
        *v = (0..n_blobs).fold(background, |acc, i| {
            acc + gaussian_contribution(
                p.as_array_view(),
                centers.slice(s![i, ..]),
                radii[i],
                intensities[i],
                falloffs[i].to_f64(),
            )
        });
    });

    Ok(blobs_arr)
}

pub fn logistic_metaballs<'a, T, A, B>(
    centers: A,
    radii: B,
    intensities: B,
    falloffs: B,
    background: T,
    shape: &[usize],
) -> Result<ArrayD<f64>, ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    B: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let centers: ArrayBase<ViewRepr<&'a T>, Ix2> = centers.into();
    let radii: ArrayBase<ViewRepr<&'a T>, Ix1> = radii.into();
    let intensities: ArrayBase<ViewRepr<&'a T>, Ix1> = intensities.into();
    let falloffs: ArrayBase<ViewRepr<&'a T>, Ix1> = falloffs.into();
    let background = background.to_f64();
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
    // TODO ensure shape length is the same as n_dims length
    let mut blobs_arr = ArrayD::from_elem(shape, background);
    blobs_arr.view_mut().indexed_iter_mut().for_each(|(p, v)| {
        *v = (0..n_blobs).fold(background, |acc, i| {
            acc.max(logistic_contribution(
                p.as_array_view(),
                centers.slice(s![i, ..]),
                radii[i],
                intensities[i],
                falloffs[i].to_f64(),
            ))
        });
    });

    Ok(blobs_arr)
}

/// TODO
fn gaussian_contribution<T>(
    current_pos: ArrayView1<usize>,
    center_pos: ArrayView1<T>,
    radius: T,
    intensity: T,
    falloff: f64,
) -> f64
where
    T: AsNumeric,
{
    let dist_sq = current_pos
        .iter()
        .zip(center_pos.iter())
        .map(|(&cur, &cen)| {
            let diff = cur as f64 - cen.to_f64();
            diff * diff
        })
        .sum::<f64>();
    let sigma_sq = (radius * radius).to_f64();

    intensity.to_f64() * (-dist_sq / (falloff * sigma_sq)).exp()
}

/// TODO
fn logistic_contribution<T>(
    current_pos: ArrayView1<usize>,
    center_pos: ArrayView1<T>,
    radius: T,
    intensity: T,
    falloff: f64,
) -> f64
where
    T: AsNumeric,
{
    let dist: f64 = current_pos
        .iter()
        .zip(center_pos.iter())
        .map(|(&cur, &cen)| {
            let diff = cur as f64 - cen.to_f64();
            diff * diff
        })
        .sum::<f64>()
        .sqrt();
    let k = falloff.max(1e-12);
    let expo = ((dist - radius.to_f64()) / k).exp();
    let soft = 1.0 / (1.0 + expo);

    intensity.to_f64() * soft
}
