use std::collections::HashMap;

use ndarray::{
    Array2, Array3, ArrayBase, ArrayView2, AsArray, Axis, Ix1, Ix3, ViewRepr, Zip, s, stack,
};
use rayon::prelude::*;

use crate::error::ImgalError;
use crate::integration::midpoint;
use crate::parameter::omega;
use crate::traits::numeric::AsNumeric;

/// Compute the real and imaginary (G, S) coordinates of a 3-dimensional decay
/// image.
///
/// # Description
///
/// Computes the real (G) and imaginary (S) components using normalized sine
/// and cosine Fourier transforms:
///
/// ```text
/// G = ∫(I(t) * cos(nωt) * dt) / ∫(I(t) * dt)
/// S = ∫(I(t) * sin(nωt) * dt) / ∫(I(t) * dt)
/// ```
///
/// # Arguments
///
/// * `data`: I(t), the decay data image.
/// * `period`: The period (_i.e._ time interval).
/// * `harmonic`: The harmonic value. If `None`, then `harmonic = 1.0`.
/// * `axis`: The decay or lifetime axis. If `None`, then `axis = 2`.
///
/// # Returns
///
/// * `Ok(Array3<f64>)`: The real and imaginary coordinates as a 3D
///   (ch, row, col) image, where G and S are indexed at `0` and `1`
///   respectively on the _channel_ axis.
/// * `Err(ImgalError)`: If `axis >= 3`.
pub fn gs_image<'a, T, A>(
    data: A,
    period: f64,
    mask: Option<ArrayView2<bool>>,
    harmonic: Option<f64>,
    axis: Option<usize>,
) -> Result<Array3<f64>, ImgalError>
where
    A: AsArray<'a, T, Ix3>,
    T: 'a + AsNumeric,
{
    let a = axis.unwrap_or(2);
    if a >= 3 {
        return Err(ImgalError::InvalidAxis {
            axis_idx: a,
            dim_len: 3,
        });
    }
    // set integral parameters, initialize the working and output buffers
    let data: ArrayBase<ViewRepr<&'a T>, Ix3> = data.into();
    let h = harmonic.unwrap_or(1.0);
    let w = omega(period);
    let n: usize = data.len_of(Axis(a));
    let dt: f64 = period / n as f64;
    let h_w_dt: f64 = h * w * dt;
    let mut w_cos_buf: Vec<f64> = Vec::with_capacity(n);
    let mut w_sin_buf: Vec<f64> = Vec::with_capacity(n);
    let mut shape = data.shape().to_vec();
    shape.remove(a);
    let mut g_arr = Array2::<f64>::zeros((shape[0], shape[1]));
    let mut s_arr = Array2::<f64>::zeros((shape[0], shape[1]));
    // load the sine and cosine waveform buffers
    for i in 0..n {
        w_cos_buf.push(f64::cos(h_w_dt * (i as f64)));
        w_sin_buf.push(f64::sin(h_w_dt * (i as f64)));
    }
    // compute the G/S phasor coordinates (optinally only within a mask region)
    // with midpoint integration
    let lanes = data.lanes(Axis(a));
    if let Some(msk) = mask {
        Zip::from(lanes)
            .and(msk)
            .and(&mut g_arr)
            .and(&mut s_arr)
            .par_for_each(|ln, m, g, s| {
                if *m {
                    let mut iv = 0.0;
                    let mut gv = 0.0;
                    let mut sv = 0.0;
                    ln.iter()
                        .zip(w_cos_buf.iter())
                        .zip(w_sin_buf.iter())
                        .for_each(|((v, cosv), sinv)| {
                            let vf: f64 = (*v).to_f64();
                            iv += vf;
                            gv += vf * cosv;
                            sv += vf * sinv;
                        });
                    iv *= dt;
                    gv *= dt;
                    sv *= dt;
                    *g = gv / iv;
                    *s = sv / iv;
                } else {
                    *g = 0.0;
                    *s = 0.0;
                }
            });
    } else {
        Zip::from(&mut g_arr)
            .and(&mut s_arr)
            .and(lanes)
            .par_for_each(|g, s, ln| {
                let mut iv = 0.0;
                let mut gv = 0.0;
                let mut sv = 0.0;
                ln.iter()
                    .zip(w_cos_buf.iter())
                    .zip(w_sin_buf.iter())
                    .for_each(|((v, cosv), sinv)| {
                        let vf: f64 = (*v).to_f64();
                        iv += vf;
                        gv += vf * cosv;
                        sv += vf * sinv;
                    });
                iv *= dt;
                gv *= dt;
                sv *= dt;
                *g = gv / iv;
                *s = sv / iv;
            });
    }
    Ok(stack(Axis(2), &[g_arr.view(), s_arr.view()]).unwrap())
}

/// Compute the real and imaginary (G, S) coordinates of a 3D decay image.
///
/// # Description
///
/// Computes real and imaginary (G, S) coordinates for a given point ROI point
/// cloud.
///
/// ```text
/// G = ∫(I(t) * cos(nωt) * dt) / ∫(I(t) * dt)
/// S = ∫(I(t) * sin(nωt) * dt) / ∫(I(t) * dt)
/// ```
///
/// # Arguments
///
/// * `data`: I(t), the decay 3D array.
/// * `period`: The period (*i.e.* time interval).
/// * `rois`: A HashMap of point clouds representing Regions of Interests
///   (ROIs). 2D ROIs are expected.
/// * `harmonic`: The harmonic value. If `None`, then `harmonic = 1.0`.
/// * `axis`: The decay or lifetime axis. If `None`, then `axis = 2`.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Ok(HashMap<u64, Array2<f64>>)`: A HashMap where the keys are the ROI
///   labels and values are the G and S values computed at each point in the
///   input ROI point cloud. Each computed ROI point cloud has shape `(p, 2)`,
///   where `p` is the number of points.
/// * `Err(ImgalError)`: If `axis >= 3`.
pub fn gs_map<'a, T, A>(
    data: A,
    period: f64,
    rois: &HashMap<u64, Array2<usize>>,
    harmonic: Option<f64>,
    axis: Option<usize>,
    parallel: bool,
) -> Result<HashMap<u64, Array2<f64>>, ImgalError>
where
    A: AsArray<'a, T, Ix3>,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, Ix3> = data.into();
    let axis = axis.unwrap_or(2);
    if axis >= 3 {
        return Err(ImgalError::InvalidAxis {
            axis_idx: axis,
            dim_len: 3,
        });
    }
    let vec_to_arr = |k: u64, v: Vec<Vec<f64>>| {
        let arr = Array2::from_shape_vec((v.len(), v[0].len()), v.into_iter().flatten().collect())
            .expect("Failed to reshape ROI point cloud into an Array2<f64>.");
        (k, arr)
    };
    if parallel {
        let cloud_map = rois
            .into_iter()
            .par_bridge()
            .fold(
                || HashMap::new(),
                |mut map: HashMap<u64, Vec<Vec<f64>>>, (&k, v)| {
                    let roi_coords = v.lanes(Axis(1));
                    roi_coords.into_iter().for_each(|p| {
                        let row = p[0];
                        let col = p[1];
                        let ln = match axis {
                            0 => data.slice(s![.., row, col]),
                            1 => data.slice(s![row, .., col]),
                            _ => data.slice(s![row, col, ..]),
                        };
                        let g = real_coord(&ln, period, harmonic);
                        let s = imaginary_coord(&ln, period, harmonic);
                        map.entry(k).or_insert_with(Vec::new).push(vec![g, s]);
                    });
                    map
                },
            )
            .reduce(
                || HashMap::new(),
                |mut map_a, map_b| {
                    map_b.into_iter().for_each(|(k, mut v)| {
                        map_a.entry(k).or_insert_with(Vec::new).append(&mut v);
                    });
                    map_a
                },
            );
        Ok(cloud_map
            .into_iter()
            .map(|(k, v)| vec_to_arr(k, v))
            .collect())
    } else {
        let mut cloud_map: HashMap<u64, Vec<Vec<f64>>> = HashMap::new();
        rois.into_iter().for_each(|(&k, v)| {
            let roi_coords = v.lanes(Axis(1));
            roi_coords.into_iter().for_each(|p| {
                let row = p[0];
                let col = p[1];
                let ln = match axis {
                    0 => data.slice(s![.., row, col]),
                    1 => data.slice(s![row, .., col]),
                    _ => data.slice(s![row, col, ..]),
                };
                let g = real_coord(&ln, period, harmonic);
                let s = imaginary_coord(&ln, period, harmonic);
                cloud_map.entry(k).or_insert_with(Vec::new).push(vec![g, s]);
            });
        });
        Ok(cloud_map
            .into_iter()
            .map(|(k, v)| vec_to_arr(k, v))
            .collect())
    }
}

/// Compute the imaginary (S) component of a 1-dimensional decay curve.
///
/// # Description
///
/// Computes the imaginary (S) component is calculated using the normalized sine
/// Fourier transform:
///
/// ```text
/// S = ∫(I(t) * sin(nωt) * dt) / ∫(I(t) * dt)
/// ```
///
/// Where `n` and `ω` are harmonic and omega values respectively.
///
/// # Arguments
///
/// * `data`: I(t), the 1-dimensonal decay curve.
/// * `period`: The period (_i.e._ time interval).
/// * `harmonic`: The harmonic value. If `None`, then `harmonic = 1.0`.
///
/// # Returns
///
/// * `f64`: The imaginary component, S.
pub fn imaginary_coord<'a, T, A>(data: A, period: f64, harmonic: Option<f64>) -> f64
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, Ix1> = data.into();
    let h: f64 = harmonic.unwrap_or(1.0);
    let w: f64 = omega(period);
    let n: usize = data.len();
    let dt: f64 = period / (n as f64);
    let h_w_dt: f64 = h * w * dt;
    let mut buf = Vec::with_capacity(n);
    for i in 0..n {
        buf.push(data[i].to_f64() * f64::sin(h_w_dt * (i as f64)));
    }
    let i_sin_integral: f64 = midpoint(&buf, Some(dt), false);
    let i_integral: f64 = midpoint(data, Some(dt), false);
    i_sin_integral / i_integral
}

/// Compute the real (G) component of a 1-dimensional decay curve.
///
/// # Description
///
/// Computes the real (G) component is calculated using the normalized cosine
/// Fourier transform:
///
/// ```text
/// G = ∫(I(t) * cos(nωt) * dt) / ∫(I(t) * dt)
/// ```
///
/// Where `n` and `ω` are harmonic and omega values respectively.
///
/// # Arguments
///
/// * `data`: I(t), the 1-dimensional decay curve.
/// * `period`: The period, (_i.e._ time interval).
/// * `harmonic`: The harmonic value. If `None`, then `harmonic = 1.0`.
///
/// # Returns
///
/// * `f64`: The real component, G.
pub fn real_coord<'a, T, A>(data: A, period: f64, harmonic: Option<f64>) -> f64
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, Ix1> = data.into();
    let h: f64 = harmonic.unwrap_or(1.0);
    let w: f64 = omega(period);
    let n: usize = data.len();
    let dt: f64 = period / (n as f64);
    let h_w_dt: f64 = h * w * dt;
    let mut buf = Vec::with_capacity(n);
    for i in 0..n {
        buf.push(data[i].to_f64() * f64::cos(h_w_dt * (i as f64)));
    }
    let i_cos_integral: f64 = midpoint(&buf, Some(dt), false);
    let i_integral: f64 = midpoint(data, Some(dt), false);
    i_cos_integral / i_integral
}
