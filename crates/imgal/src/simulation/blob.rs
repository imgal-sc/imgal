use ndarray::{ArrayBase, ArrayD, ArrayView1, AsArray, Dimension, Ix1, Ix2, ViewRepr, s};

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Create an n-dimensional Gaussian metaballs image.
///
/// # Description
///
/// Creates a simulated n-dimensional blobs image using a variant of Jim Blinn's
/// metaballs blob simulation algorithm. Metaballs are n-dimensional blob
/// isosurfaces that are able to interact with each other. This function uses a
/// Gaussian falloff strategy to simulate a smooth and continuous blob border
/// with no sharp edges.
///
/// # Arguments
///
/// * `centers`: A 2D array with `(p, D)`, where `p` is the number of blobs and
///   `D` is the number of dimensions.
/// * `radii`: A 1D array where each element represents a blob radius.
/// * `intensities`: A 1D array where each element represents a blob intensity.
/// * `falloffs`: A 1D array where each element represents the "falloff" value
///   for a given blob that controls the rate of intensity decay from the blob
///   center. High values result in a more blured border effect and low values
///   have a more defined border.
/// * `background`: The background intensity value for the image.
/// * `shape`: The shape of the output n-dimensional array.
///
/// # Returns
///
/// * `Ok(ArrayD<f64>)`: An n-dimensional array containing the metaballs blob
///   simulation, where each pixel value is the *sum* of Gaussian contributions
///   from each blob and the background.
/// * `Err(ImgalError)`: If the number of blobs and `radii.len()` or
///   `intensities.len()` do not match.
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

/// Create an n-dimensional logistic metaballs image.
///
/// # Description
///
/// Creates a simulated n-dimensional blobs image using a variant of Jim Blinn's
/// metaballs blob simulation algorithm. Metaballs are n-dimensional blob
/// isosurfaces that are able to interact with each other. This function uses a
/// logistic (sigmoid) falloff function to simulate smooth and crisp blob
/// borders. Logistic metaballs, unlike traditional metaballs, do not fuse
/// together but instead deform against neighboring blobs.
///
/// # Arguments
///
/// * `centers`: A 2D array with `(p, D)`, where `p` is the number of blobs and
///   `D` is the number of dimensions.
/// * `radii`: A 1D array where each element represents a blob radius.
/// * `intensities`: A 1D array where each element represents a blob intensity.
/// * `falloffs`: A 1D array where each element represents the "falloff" value
///   for a given blob that controls the value transition steepness from the
///   center of the blob to the edge. High values result in longer transitions
///   to the background, creating larger or inflated blobs. Low values result in
///   short or rapid transitions to the background, creating crisp edges.
/// * `background`: The background intensity value for the image.
/// * `shape`: The shape of the output n-dimensional array.
///
/// # Returns
///
/// * `Ok(ArrayD<f64>)`: An n-dimensional array containing the metaballs blob
///   simulation, where each pixel value is the *maximum* contribution of any
///   blob at that position.
/// * `Err(ImgalError)`: If the number of blobs and `radii.len()` or
///   `intensities.len()` do not match.
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

/// Compute the Gaussian contribution of a single blob at a given position.
///
/// # Arguments
///
/// * `current_pos`: The current pixel coordinates as a 1D array view.
/// * `center_pos`: The given blob's center coordinates as a 1D array view.
/// * `radius`: The radius of the given blob.
/// * `intensity`: The intensity of the given blob.
/// * `falloff`: The Gaussian falloff value.
///
/// # Returns
///
/// * `f64`: The Gaussian intensity contribution of the given blob and position.
#[inline]
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

/// Compute the logistic function contribution of a single blob at a given
/// position.
///
/// # Arguments
///
/// * `current_pos`: The current pixel coordinates at a 1D array view.
/// * `center_pos`: The given blob's center coordinates as a 1D array view.
/// * `radius`: The radius of the given blob.
/// * `intensity`: The intensity of the given blob.
/// * `falloff`: The logistic function contribution of the given blob and
///   position.
///
/// # Returns
///
/// * `f64`: The logistic function intensity contribution of the given blob and
///   position.
#[inline]
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
