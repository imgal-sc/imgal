use ndarray::{
    Array2, Array3, ArrayBase, ArrayView2, AsArray, Axis, Ix1, Ix3, ViewRepr, Zip, stack,
};

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
    // create a view of the data
    let view: ArrayBase<ViewRepr<&'a T>, Ix3> = data.into();

    // set optional parameters if needed
    let h = harmonic.unwrap_or(1.0);
    let a = axis.unwrap_or(2);

    // check if axis parameter is valid
    if a >= 3 {
        return Err(ImgalError::InvalidAxis {
            axis_idx: a,
            dim_len: 3,
        });
    }

    // initialize phasor parameters
    let w = omega(period);
    let n: usize = view.len_of(Axis(a));
    let dt: f64 = period / n as f64;
    let h_w_dt: f64 = h * w * dt;

    // initialize buffers
    let mut w_cos_buf: Vec<f64> = Vec::with_capacity(n);
    let mut w_sin_buf: Vec<f64> = Vec::with_capacity(n);

    // drop specified axis and create new G and S output arrays with new shape
    let mut shape = view.shape().to_vec();
    shape.remove(a);
    let mut g_arr = Array2::<f64>::zeros((shape[0], shape[1]));
    let mut s_arr = Array2::<f64>::zeros((shape[0], shape[1]));

    // load the waveform buffers
    for i in 0..n {
        w_cos_buf.push(f64::cos(h_w_dt * (i as f64)));
        w_sin_buf.push(f64::sin(h_w_dt * (i as f64)));
    }

    // compute phasor coordinates per lane, optionally only in mask area
    let lanes = view.lanes(Axis(a));
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
                            // midpoint integration
                            let vf: f64 = (*v).to_f64();
                            iv += vf;
                            gv += vf * cosv;
                            sv += vf * sinv;
                        });
                    // midpoint integration, multiply by data point width
                    iv *= dt;
                    gv *= dt;
                    sv *= dt;
                    // normalize G/S values and write to output arrays
                    *g = gv / iv;
                    *s = sv / iv;
                } else {
                    // if false on mask, set G/S output to zero
                    *g = 0.0;
                    *s = 0.0;
                }
            });
    } else {
        // compute phasor coordinates per lane in the entire array, no mask
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
                        // midpoint integration
                        let vf: f64 = (*v).to_f64();
                        iv += vf;
                        gv += vf * cosv;
                        sv += vf * sinv;
                    });
                // midpoint integration, multiply by data point width
                iv *= dt;
                gv *= dt;
                sv *= dt;
                // normalize G/S values and write to output arrays
                *g = gv / iv;
                *s = sv / iv;
            });
    }

    // stack G and S arrays, (row, col, ch)
    Ok(stack(Axis(2), &[g_arr.view(), s_arr.view()]).unwrap())
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
pub fn imaginary_coordinate<'a, T, A>(data: A, period: f64, harmonic: Option<f64>) -> f64
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    // create a view of the data
    let view: ArrayBase<ViewRepr<&'a T>, Ix1> = data.into();

    // set optional parameters if needed
    let h: f64 = harmonic.unwrap_or(1.0);
    let w: f64 = omega(period);

    // integrate sine transform (imaginary)
    let n: usize = view.len();
    let dt: f64 = period / (n as f64);
    let h_w_dt: f64 = h * w * dt;
    let mut buf = Vec::with_capacity(n);
    for i in 0..n {
        buf.push(view[i].to_f64() * f64::sin(h_w_dt * (i as f64)));
    }
    let i_sin_integral: f64 = midpoint(&buf, Some(dt));
    let i_integral: f64 = midpoint(view, Some(dt));

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
pub fn real_coordinate<'a, T, A>(data: A, period: f64, harmonic: Option<f64>) -> f64
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    // create a view of the data
    let view: ArrayBase<ViewRepr<&'a T>, Ix1> = data.into();

    // set optional parameters if needed
    let h: f64 = harmonic.unwrap_or(1.0);
    let w: f64 = omega(period);

    // integrate cosine transform (real)
    let n: usize = view.len();
    let dt: f64 = period / (n as f64);
    let h_w_dt: f64 = h * w * dt;
    let mut buf = Vec::with_capacity(n);
    for i in 0..n {
        buf.push(view[i].to_f64() * f64::cos(h_w_dt * (i as f64)));
    }
    let i_cos_integral: f64 = midpoint(&buf, Some(dt));
    let i_integral: f64 = midpoint(view, Some(dt));

    i_cos_integral / i_integral
}
