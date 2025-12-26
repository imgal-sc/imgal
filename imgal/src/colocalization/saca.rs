use std::mem;

use ndarray::{
    Array2, Array3, Array4, ArrayBase, ArrayD, ArrayView2, ArrayView3, ArrayViewMut2,
    ArrayViewMut3, ArrayViewMut4, AsArray, Axis, Dimension, Ix2, Ix3, ViewRepr, Zip,
};
use rayon::prelude::*;

use crate::distribution::inverse_normal_cdf;
use crate::error::ImgalError;
use crate::kernel::neighborhood::{weighted_circle_kernel, weighted_sphere_kernel};
use crate::statistics::{effective_sample_size, weighted_kendall_tau_b};
use crate::threshold::manual_mask;
use crate::traits::numeric::AsNumeric;

/// Compute 2-dimensional colocalization strength with Spatially Adaptive
/// Colocalization Analysis (SACA).
///
/// # Description
///
/// Computes a pixel-wise _z-score_ indicating colocalization and
/// anti-colocalization strength on 2-dimensional input images using the
/// Spatially Adaptive Colocalization Analysis (SACA) framework. Per pixel SACA
/// utilizes a propagation and separation strategy to adaptively expand a
/// weighted circular kernel that defines the pixel of consideration's
/// neighborhood. The pixels within the neighborhood are assigned weights based
/// on their distance from the center pixel (decreasing with distance), ranked
/// and their colocalization coefficient computed using Kendall's Tau-b rank
/// correlation.
///
/// # Arguments
///
/// * `data_a`: A 2-dimensional input image to measure colocalization strength,
///   with the same shape as `data_b`.
/// * `data_b`: A 2-dimensional input image to measure colocalization strength,
///   with the same shape as `data_a`.
/// * `threshold_a`: Pixel intensity threshold value for `data_a`. Pixels below
///   this value are given a weight of `0.0` if the pixel is in the circular
///   neighborhood.
/// * `threshold_b`: Pixel intensity threshold value for `data_b`. Pixels below
///   this value are given a weight of `0.0` if the pixel is in the circular
///   neighborhood.
///
/// # Returns
///
/// * `OK(Array2<f64>)`: The pixel-wise _z-score_ indicating colocalization or
///   anti-colocalization by its sign and the degree or strength of the
///   relationship through its absolute values.
/// * `Err(ImgalError)`: If the dimensions of image `data_a` and `data_b` do not
///   match.
///
/// # Reference
///
/// <https://doi.org/10.1109/TIP.2019.2909194>
pub fn saca_2d<'a, T, A>(
    data_a: A,
    data_b: A,
    threshold_a: T,
    threshold_b: T,
) -> Result<Array2<f64>, ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    let view_a: ArrayBase<ViewRepr<&'a T>, Ix2> = data_a.into();
    let view_b: ArrayBase<ViewRepr<&'a T>, Ix2> = data_b.into();

    // validate input images have the same shape
    let dims_a = view_a.dim();
    let dims_b = view_b.dim();
    if dims_a != dims_b {
        return Err(ImgalError::MismatchedArrayShapes {
            shape_a: vec![dims_a.0, dims_a.1],
            shape_b: vec![dims_b.0, dims_b.1],
        });
    }

    // create kendall tau b working buffers and output container
    let mut result = Array2::<f64>::zeros(dims_a);
    let mut new_tau = Array2::<f64>::zeros(dims_a);
    let mut new_sqrt_n = Array2::<f64>::zeros(dims_a);
    let mut old_tau = Array2::<f64>::zeros(dims_a);
    let mut old_sqrt_n = Array2::<f64>::ones(dims_a);
    let mut stop = Array3::<f64>::zeros((dims_a.0, dims_a.1, 3));

    // set up saca parameters, see reference on "dn" value selection for lambda
    let dn = ((dims_a.0 * dims_a.1) as f64).ln().sqrt() * 2.0;
    let lambda = dn * 1.0;
    let tu: usize = 15;
    let tl: usize = 8;
    let mut size_f: f64 = 1.0;
    let mut radius: usize = 1;
    let step_size: f64 = 1.15;
    let mut lower_bound_check = false;

    // run the multiscale adaptive analysis
    (0..tu).for_each(|s| {
        radius = size_f.floor() as usize;
        single_iteration_2d(
            view_a,
            view_b,
            threshold_a,
            threshold_b,
            result.view_mut(),
            new_tau.view_mut(),
            new_sqrt_n.view_mut(),
            stop.view_mut(),
            old_tau.view_mut(),
            old_sqrt_n.view_mut(),
            radius,
            dn,
            lambda,
            lower_bound_check,
        );
        mem::swap(&mut old_tau, &mut new_tau);
        mem::swap(&mut old_sqrt_n, &mut new_sqrt_n);
        size_f *= step_size;
        if s == tl {
            lower_bound_check = true;
            let lanes = stop.lanes_mut(Axis(2));
            Zip::from(lanes)
                .and(new_tau.view())
                .and(new_sqrt_n.view())
                .par_for_each(|mut ln, nt, ns| {
                    ln[1] = *nt;
                    ln[2] = *ns;
                });
        }
    });

    Ok(result)
}

/// Compute 3-dimensional colocalization strength with Spatially Adaptive
/// Colocalization Analysis (SACA).
///
/// # Description
///
/// Computes a pixel-wise _z-score_ indicating colocalization and
/// anti-colocalization strength on 3-dimensional input images using the
/// Spatially Adaptive Colocalization Analysis (SACA) framework. Per pixel SACA
/// utilizes a propagation and separation strategy to adaptively expand a
/// weighted spherical kernel that defines the pixel of consideration's
/// neighborhood. The pixels within the neighborhood are assigned weights based
/// on their distance from the center pixel (decreasing with distance), ranked
/// and their colocalization coefficient computed using Kendall's Tau-b rank
/// correlation.
///
/// # Arguments
///
/// * `data_a`: A 3-dimensional input image to measure colocalization strength,
///   with the same shape as `data_b`.
/// * `data_b`: A 3-dimensional input image to measure colocalization strength,
///   with the same shape as `data_a`.
/// * `threshold_a`: Pixel intensity threshold value for `data_a`. Pixels below
///   this value are given a weight of `0.0` if the pixel is in the circular
///   neighborhood.
/// * `threshold_b`: Pixel intensity threshold value for `data_b`. Pixels below
///   this value are given a weight of `0.0` if the pixel is in the circular
///   neighborhood.
///
/// # Returns
///
/// * `OK(Array3<f64>)`: The pixel-wise _z-score_ indicating colocalization or
///   anti-colocalization by its sign and the degree or strength of the
///   relationship through its absolute values.
/// * `Err(ImgalError)`: If the dimensions of image `data_a` and `data_b` do not
///   match.
///
/// # Reference
///
/// <https://doi.org/10.1109/TIP.2019.2909194>
pub fn saca_3d<'a, T, A>(
    data_a: A,
    data_b: A,
    threshold_a: T,
    threshold_b: T,
) -> Result<Array3<f64>, ImgalError>
where
    A: AsArray<'a, T, Ix3>,
    T: 'a + AsNumeric,
{
    let view_a: ArrayBase<ViewRepr<&'a T>, Ix3> = data_a.into();
    let view_b: ArrayBase<ViewRepr<&'a T>, Ix3> = data_b.into();

    // validate input images have the same shape
    let dims_a = view_a.dim();
    let dims_b = view_a.dim();
    if dims_a != dims_b {
        return Err(ImgalError::MismatchedArrayShapes {
            shape_a: vec![dims_a.0, dims_a.1, dims_a.2],
            shape_b: vec![dims_b.0, dims_b.1, dims_b.2],
        });
    }

    // create kendall tau b working buffers and output container
    let mut result = Array3::<f64>::zeros(dims_a);
    let mut new_tau = Array3::<f64>::zeros(dims_a);
    let mut new_sqrt_n = Array3::<f64>::zeros(dims_a);
    let mut old_tau = Array3::<f64>::zeros(dims_a);
    let mut old_sqrt_n = Array3::<f64>::ones(dims_a);
    let mut stop = Array4::<f64>::zeros((dims_a.0, dims_a.1, dims_a.2, 3));

    // set up saca parameters, see reference on "dn" value selection for lambda
    let dn = ((dims_a.0 * dims_a.1 * dims_a.2) as f64).ln().sqrt() * 2.0;
    let lambda = dn * 1.0;
    let tu: usize = 15;
    let tl: usize = 8;
    let mut size_f: f64 = 1.0;
    let mut radius: usize = 1;
    let step_size: f64 = 1.15;
    let mut lower_bound_check = false;

    // run the multiscale adaptive analysis
    (0..tu).for_each(|s| {
        radius = size_f.floor() as usize;
        single_iteration_3d(
            view_a,
            view_b,
            threshold_a,
            threshold_b,
            result.view_mut(),
            new_tau.view_mut(),
            new_sqrt_n.view_mut(),
            stop.view_mut(),
            old_tau.view_mut(),
            old_sqrt_n.view_mut(),
            radius,
            dn,
            lambda,
            lower_bound_check,
        );
        mem::swap(&mut old_tau, &mut new_tau);
        mem::swap(&mut old_sqrt_n, &mut new_sqrt_n);
        size_f *= step_size;
        if s == tl {
            lower_bound_check = true;
            let lanes = stop.lanes_mut(Axis(3));
            Zip::from(lanes)
                .and(new_tau.view())
                .and(new_sqrt_n.view())
                .par_for_each(|mut ln, nt, ns| {
                    ln[1] = *nt;
                    ln[2] = *ns;
                });
        }
    });

    Ok(result)
}

/// Create a significant pixel mask from a pixel-wise _z-score_ array.
///
/// # Description
///
/// Creates a boolean array representing significant pixels (_i.e._ the mask) by
/// applying Bonferroni correction to adjust for multiple comparisons.
///
/// # Arguments
///
/// * `data`: The pixel-wise _z-score_ indicating colocalization or
///   anti-colocalization strength.
/// * `alpha`: The significance level representing the maximum type I error
///   (_i.e._ false positive error) allowed (default = 0.05).
///
/// # Returns
///
/// * `ArrayD<bool>`: The significant pixel mask where `true` pixels represent
///   significant _z-score_ values.
///
/// # Reference
///
/// <https://doi.org/10.1109/TIP.2019.2909194>
pub fn saca_significance_mask<'a, A, D>(data: A, alpha: Option<f64>) -> ArrayD<bool>
where
    A: AsArray<'a, f64, D>,
    D: Dimension,
{
    let view: ArrayBase<ViewRepr<&'a f64>, D> = data.into();
    let alpha = alpha.unwrap_or(0.05);
    let q = inverse_normal_cdf(1.0 - (alpha / view.len() as f64)).unwrap();

    manual_mask(&view, q)
}

/// Fill working buffers from 2-dimensional data.
fn fill_buffers_2d<T>(
    data_a: ArrayView2<T>,
    data_b: ArrayView2<T>,
    kernel: ArrayView2<f64>,
    old_tau: ArrayView2<f64>,
    old_sqrt_n: ArrayView2<f64>,
    buf_a: &mut [T],
    buf_b: &mut [T],
    buf_w: &mut [f64],
    dn: f64,
    radius: usize,
    pos_row: usize,
    pos_col: usize,
    buf_row_start: usize,
    buf_row_end: usize,
    buf_col_start: usize,
    buf_col_end: usize,
) where
    T: AsNumeric,
{
    // set compute parameters
    let mut i: usize = 0;
    let ot = old_tau[[pos_row, pos_col]];
    let on = old_sqrt_n[[pos_row, pos_col]];
    let on_dn = on / dn;
    let pos_row = pos_row as isize;
    let pos_col = pos_col as isize;
    let radius = radius as isize;
    let row_offset = radius - pos_row;
    let col_offset = radius - pos_col;

    // create a 2D iterator centered with the kernel
    (buf_row_start..=buf_row_end)
        .flat_map(|r| (buf_col_start..=buf_col_end).map(move |c| (r, c)))
        .for_each(|(r, c)| {
            // subtract current position to get offset from kernel center
            let kr = (r as isize + row_offset) as usize;
            let kc = (c as isize + col_offset) as usize;
            // load the buffers with data from images and associated weights
            buf_a[i] = data_a[[r, c]];
            buf_b[i] = data_b[[r, c]];
            let tau_diff_abs = (old_tau[[r, c]] - ot).abs() * on_dn;
            let w = kernel[[kr, kc]];
            buf_w[i] = if tau_diff_abs < 1.0 {
                w * (1.0 - tau_diff_abs).powi(2)
            } else {
                0.0
            };
            i += 1;
        });

    // zero out the rest of the buffers
    buf_a[i..].fill(T::default());
    buf_b[i..].fill(T::default());
    buf_w[i..].fill(0.0);
}

/// Fill working buffers from 3-dimensional data.
fn fill_buffers_3d<T>(
    data_a: ArrayView3<T>,
    data_b: ArrayView3<T>,
    kernel: ArrayView3<f64>,
    old_tau: ArrayView3<f64>,
    old_sqrt_n: ArrayView3<f64>,
    buf_a: &mut [T],
    buf_b: &mut [T],
    buf_w: &mut [f64],
    dn: f64,
    radius: usize,
    pos_pln: usize,
    pos_row: usize,
    pos_col: usize,
    buf_pln_start: usize,
    buf_pln_end: usize,
    buf_row_start: usize,
    buf_row_end: usize,
    buf_col_start: usize,
    buf_col_end: usize,
) where
    T: AsNumeric,
{
    // set compute parameters
    let mut i: usize = 0;
    let ot = old_tau[[pos_pln, pos_row, pos_col]];
    let on = old_sqrt_n[[pos_pln, pos_row, pos_col]];
    let on_dn = on / dn;
    let pos_pln = pos_pln as isize;
    let pos_row = pos_row as isize;
    let pos_col = pos_col as isize;
    let radius = radius as isize;
    let pln_offset = radius - pos_pln;
    let row_offset = radius - pos_row;
    let col_offset = radius - pos_col;

    // create a 3D iterator centered with the kernel
    (buf_pln_start..=buf_pln_end)
        .flat_map(|p| {
            (buf_row_start..=buf_row_end)
                .flat_map(move |r| (buf_col_start..=buf_col_end).map(move |c| (p, r, c)))
        })
        .for_each(|(p, r, c)| {
            // subtract current position to get offset from kernel center
            let kp = (p as isize + pln_offset) as usize;
            let kr = (r as isize + row_offset) as usize;
            let kc = (c as isize + col_offset) as usize;
            // load the buffers with data from images and associated weights
            buf_a[i] = data_a[[p, r, c]];
            buf_b[i] = data_b[[p, r, c]];
            let tau_diff_abs = (old_tau[[p, r, c]] - ot).abs() * on_dn;
            let w = kernel[[kp, kr, kc]];
            buf_w[i] = if tau_diff_abs < 1.0 {
                w * (1.0 - tau_diff_abs).powi(2)
            } else {
                0.0
            };
            i += 1;
        });

    // zero out the rest of the buffers
    buf_a[i..].fill(T::default());
    buf_b[i..].fill(T::default());
    buf_w[i..].fill(0.0);
}

/// Get the end position for filling the buffers along an axis.
fn get_end_position(location: usize, radius: usize, boundary: usize) -> usize {
    let end = location + radius;
    if end >= boundary { boundary - 1 } else { end }
}

/// Get the start position for filling the buffers along an axis.
fn get_start_position(location: usize, radius: usize) -> usize {
    if location < radius {
        0
    } else {
        location - radius
    }
}

/// Single 2-dimensional SACA iteration.
fn single_iteration_2d<T>(
    data_a: ArrayView2<T>,
    data_b: ArrayView2<T>,
    threshold_a: T,
    threshold_b: T,
    mut result: ArrayViewMut2<f64>,
    mut new_tau: ArrayViewMut2<f64>,
    mut new_sqrt_n: ArrayViewMut2<f64>,
    mut stop: ArrayViewMut3<f64>,
    old_tau: ArrayViewMut2<f64>,
    old_sqrt_n: ArrayViewMut2<f64>,
    radius: usize,
    dn: f64,
    lambda: f64,
    bound_check: bool,
) where
    T: AsNumeric,
{
    let falloff = radius as f64 * (2.5_f64).sqrt();
    let kernel = weighted_circle_kernel(radius, falloff, None).unwrap();

    // compute weighted kendall's tau and write to output
    let d = 2 * radius + 1;
    let buf_size = d * d;
    let dims_a = data_a.dim();
    let lanes = stop.lanes_mut(Axis(2));
    result
        .indexed_iter_mut()
        .zip(new_tau.iter_mut())
        .zip(new_sqrt_n.iter_mut())
        .zip(lanes)
        .par_bridge()
        .for_each(|(((((row, col), re), nt), nn), mut ln)| {
            // check stop condition and skip loop if true
            if bound_check {
                if ln[0] != 0.0 {
                    return;
                }
            }
            let tau_diff: f64;
            // create buffers for the current local neighborhood
            let mut buf_a = vec![T::default(); buf_size];
            let mut buf_b = vec![T::default(); buf_size];
            let mut buf_w = vec![0.0_f64; buf_size];
            // get the start and end positions to fill buffers
            let buf_row_start = get_start_position(row, radius);
            let buf_row_end = get_end_position(row, radius, dims_a.0);
            let buf_col_start = get_start_position(col, radius);
            let buf_col_end = get_end_position(col, radius, dims_a.1);
            fill_buffers_2d(
                data_a,
                data_b,
                kernel.view(),
                old_tau.view(),
                old_sqrt_n.view(),
                &mut buf_a,
                &mut buf_b,
                &mut buf_w,
                dn,
                radius,
                row,
                col,
                buf_row_start,
                buf_row_end,
                buf_col_start,
                buf_col_end,
            );
            buf_a
                .iter()
                .zip(buf_b.iter())
                .zip(buf_w.iter_mut())
                .for_each(|((&a, &b), w)| {
                    if a < threshold_a || b < threshold_b {
                        *w = 0.0;
                    }
                });
            *nn = effective_sample_size(&buf_w).sqrt();
            if *nn <= 0.0 {
                *nt = 0.0;
                *re = 0.0;
            } else {
                let tau = weighted_kendall_tau_b(&buf_a, &buf_b, &buf_w).unwrap_or(0.0);
                *nt = tau;
                *re = tau * *nn * 1.5;
            }
            if bound_check {
                tau_diff = (ln[1] - *nt).abs() * ln[2];
                if tau_diff > lambda {
                    ln[0] = 1.0;
                    *nt = old_tau[[row, col]];
                    *nn = old_sqrt_n[[row, col]];
                }
            }
        });
}

/// Single 3-dimensional SACA iteration.
fn single_iteration_3d<T>(
    data_a: ArrayView3<T>,
    data_b: ArrayView3<T>,
    threshold_a: T,
    threshold_b: T,
    mut result: ArrayViewMut3<f64>,
    mut new_tau: ArrayViewMut3<f64>,
    mut new_sqrt_n: ArrayViewMut3<f64>,
    mut stop: ArrayViewMut4<f64>,
    old_tau: ArrayViewMut3<f64>,
    old_sqrt_n: ArrayViewMut3<f64>,
    radius: usize,
    dn: f64,
    lambda: f64,
    bound_check: bool,
) where
    T: AsNumeric,
{
    let falloff = radius as f64 * (2.5_f64).sqrt();
    let kernel = weighted_sphere_kernel(radius, falloff, None).unwrap();

    // compute weighted kendall's tau and write to output
    let d = 2 * radius + 1;
    let buf_size = d * d * d;
    let dims_a = data_a.dim();
    let lanes = stop.lanes_mut(Axis(3));
    result
        .indexed_iter_mut()
        .zip(new_tau.iter_mut())
        .zip(new_sqrt_n.iter_mut())
        .zip(lanes)
        .par_bridge()
        .for_each(|(((((pln, row, col), re), nt), nn), mut ln)| {
            // check stop condition and skip loop if true
            if bound_check {
                if ln[0] != 0.0 {
                    return;
                }
            }
            let tau_diff: f64;
            // create buffers for the current local neighborhood
            let mut buf_a = vec![T::default(); buf_size];
            let mut buf_b = vec![T::default(); buf_size];
            let mut buf_w = vec![0.0_f64; buf_size];
            // get the start and end positions to fill buffers
            let buf_pln_start = get_start_position(pln, radius);
            let buf_pln_end = get_end_position(pln, radius, dims_a.0);
            let buf_row_start = get_start_position(row, radius);
            let buf_row_end = get_end_position(row, radius, dims_a.1);
            let buf_col_start = get_start_position(col, radius);
            let buf_col_end = get_end_position(col, radius, dims_a.2);
            fill_buffers_3d(
                data_a,
                data_b,
                kernel.view(),
                old_tau.view(),
                old_sqrt_n.view(),
                &mut buf_a,
                &mut buf_b,
                &mut buf_w,
                dn,
                radius,
                pln,
                row,
                col,
                buf_pln_start,
                buf_pln_end,
                buf_row_start,
                buf_row_end,
                buf_col_start,
                buf_col_end,
            );
            buf_a
                .iter()
                .zip(buf_b.iter())
                .zip(buf_w.iter_mut())
                .for_each(|((&a, &b), w)| {
                    if a < threshold_a || b < threshold_b {
                        *w = 0.0;
                    }
                });
            *nn = effective_sample_size(&buf_w).sqrt();
            if *nn <= 0.0 {
                *nt = 0.0;
                *re = 0.0;
            } else {
                let tau = weighted_kendall_tau_b(&buf_a, &buf_b, &buf_w).unwrap_or(0.0);
                *nt = tau;
                *re = tau * *nn * 1.5;
            }
            if bound_check {
                tau_diff = (ln[1] - *nt).abs() * ln[2];
                if tau_diff > lambda {
                    ln[0] = 1.0;
                    *nt = old_tau[[pln, row, col]];
                    *nn = old_sqrt_n[[pln, row, col]];
                }
            }
        });
}
