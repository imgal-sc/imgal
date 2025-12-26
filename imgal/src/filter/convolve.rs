use ndarray::{ArrayBase, AsArray, Ix1, ViewRepr};
use rustfft::{FftPlanner, num_complex::Complex, num_traits::Zero};

use crate::traits::numeric::AsNumeric;

/// Convolve two 1-dimensional signals using the Fast Fourier Transform (FFT).
///
/// # Description
///
/// Computes the convolution of two discrete signals (`data_a` and `data_b`) by
/// transforming them into the frequency domain, multiplying them, and then
/// transforming the result back into a signal. This function uses "same-length"
/// trimming with the first parameter `data_a`. This means that the returned
/// convolution's array length will have the same length as `data_a`.
///
/// # Arguments
///
/// * `data_a`: The first input signal to FFT convolve. Returned convolution
///   arrays will be "same-length" trimmed to `data_a`'s length.
/// * `data_b`: The second input signal to FFT convolve.
///
/// # Returns
///
/// * `Array1<f64>`: The FFT convolved result of the same length as input signal
///   `data_a`.
pub fn fft_convolve_1d<'a, T, A>(data_a: A, data_b: A) -> Vec<f64>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let view_a: ArrayBase<ViewRepr<&'a T>, Ix1> = data_a.into();
    let view_b: ArrayBase<ViewRepr<&'a T>, Ix1> = data_b.into();

    // compute FFT size, allocate sized buffers and fill with input data
    let n_a = view_a.len();
    let n_b = view_b.len();
    let n_fft = n_a + n_b - 1;
    let fft_size = n_fft.next_power_of_two();
    let mut a_fft_buf = vec![Complex::zero(); fft_size];
    let mut b_fft_buf = vec![Complex::zero(); fft_size];
    a_fft_buf[..n_a].iter_mut().enumerate().for_each(|(i, v)| {
        *v = Complex::new(view_a[i].to_f64(), 0.0);
    });
    b_fft_buf[..n_b].iter_mut().enumerate().for_each(|(i, v)| {
        *v = Complex::new(view_b[i].to_f64(), 0.0);
    });

    // create FFT planner and compute forward FFTs
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);
    fft.process(&mut a_fft_buf);
    fft.process(&mut b_fft_buf);

    // multiply in the frequency domain and extract the real component (scaled
    // and input length trimmed)
    a_fft_buf.iter_mut().enumerate().for_each(|(i, v)| {
        *v *= b_fft_buf[i];
    });
    ifft.process(&mut a_fft_buf);
    let scale = 1.0 / fft_size as f64;
    let mut result = vec![0.0; n_a];
    result.iter_mut().enumerate().for_each(|(i, v)| {
        *v = a_fft_buf[i].re * scale;
    });

    result
}

/// Deconvolve two 1-dimensional signals using the Fast Fourier Transform (FFT).
///
/// # Description
///
/// Computes the deconvolution of two discrete signals (`data_a` and `data_b`)
/// by transforming them into the frequency domain, dividing them, and then
/// transforming the result back into a signal. This function uses "same-length"
/// trimming with the first parameter `data_a`. This means that the returned
/// deconvolution's array length will have the same length as `data_a`.
///
/// # Arguments
///
/// * `data_a`: The first input signal to FFT deconvolve. Returned deconvolution
///   arrays will be "same-length" trimmed to `data_a`'s length.
/// * `data_b`: The second input singal to FFT deconvolve.
/// * `epsilon`: An epsilon value to prevent division by zero errors (default =
///   `1e-8`).
///
/// # Returns
///
/// * `ArrayView1<f64>`: The FFT deconvolved result of the same length as input
///   signal `data_a`.
pub fn fft_deconvolve_1d<'a, T, A>(data_a: A, data_b: A, epsilon: Option<f64>) -> Vec<f64>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let view_a: ArrayBase<ViewRepr<&'a T>, Ix1> = data_a.into();
    let view_b: ArrayBase<ViewRepr<&'a T>, Ix1> = data_b.into();
    let epsilon = epsilon.unwrap_or(1e-8);

    // compute FFT size, allocate sized buffers and fill with input data
    let n_a = view_a.len();
    let n_b = view_b.len();
    let n_fft = n_a + n_b - 1;
    let fft_size = n_fft.next_power_of_two();
    let mut a_fft_buf = vec![Complex::zero(); fft_size];
    let mut b_fft_buf = vec![Complex::zero(); fft_size];
    a_fft_buf[..n_a].iter_mut().enumerate().for_each(|(i, v)| {
        *v = Complex::new(view_a[i].to_f64(), 0.0);
    });
    b_fft_buf[..n_b].iter_mut().enumerate().for_each(|(i, v)| {
        *v = Complex::new(view_b[i].to_f64(), 0.0);
    });

    // create FFT planner and compute forward FFTs
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);
    fft.process(&mut a_fft_buf);
    fft.process(&mut b_fft_buf);

    // divide in the frequency domain with epsilon value and extract the real
    // component (scaled and input length trimmed)
    a_fft_buf.iter_mut().enumerate().for_each(|(i, v)| {
        if v.norm_sqr() > epsilon {
            *v /= b_fft_buf[i]
        } else {
            *v = Complex::zero();
        }
    });
    ifft.process(&mut a_fft_buf);
    let scale = 1.0 / fft_size as f64;
    let mut result = vec![0.0; n_a];
    result.iter_mut().enumerate().for_each(|(i, v)| {
        *v = a_fft_buf[i].re * scale;
    });

    result
}
