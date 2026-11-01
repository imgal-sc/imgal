use ndarray::{Array1, ArrayBase, AsArray, Ix1, ViewRepr, Zip};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex, num_traits::Zero};

use crate::prelude::*;

/// Convolve two 1D signals using the Fast Fourier Transform (FFT).
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
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `Array1<f64>`: The FFT convolved result of the same length as input signal
///   `data_a`.
pub fn fft_convolve_1d<'a, T, A>(data_a: A, data_b: A, threads: Option<usize>) -> Array1<f64>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let data_a: ArrayBase<ViewRepr<&'a T>, Ix1> = data_a.into();
    let data_b: ArrayBase<ViewRepr<&'a T>, Ix1> = data_b.into();
    let n_a = data_a.len();
    let n_b = data_b.len();
    let n_fft = n_a + n_b - 1;
    let fft_size = n_fft.next_power_of_two();
    let mut a_fft_buf = vec![Complex::zero(); fft_size];
    let mut b_fft_buf = vec![Complex::zero(); fft_size];
    let load_buffers = |a_buf: &mut Complex<f64>, b_buf: &mut Complex<f64>, a: &T, b: &T| {
        *a_buf = Complex::new(a.to_f64(), 0.0);
        *b_buf = Complex::new(b.to_f64(), 0.0);
    };
    par!(threads,
        seq_exp: Zip::from(&mut a_fft_buf[..n_a]).and(&mut b_fft_buf[..n_b])
            .and(data_a.view())
            .and(data_b.view())
            .for_each(&load_buffers),
        par_exp: Zip::from(&mut a_fft_buf[..n_a]).and(&mut b_fft_buf[..n_b])
            .and(data_a.view())
            .and(data_b.view())
            .par_for_each(&load_buffers));
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);
    fft.process(&mut a_fft_buf);
    fft.process(&mut b_fft_buf);
    // multiply in the frequency domain and extract the real component (scaled
    // and input length trimmed)
    let mul_calc = |a: &mut Complex<f64>, b: &Complex<f64>| {
        *a *= b;
    };
    par!(threads,
        seq_exp: a_fft_buf.iter_mut().zip(b_fft_buf.iter())
            .for_each(|(a, b)| mul_calc(a, b)),
        par_exp: a_fft_buf.par_iter_mut().zip(b_fft_buf.par_iter())
            .for_each(|(a, b)| mul_calc(a, b)));
    ifft.process(&mut a_fft_buf);
    let scale = 1.0 / fft_size as f64;
    par!(threads,
        seq_exp: (0..n_a).zip(a_fft_buf.iter()).map(|(_, v)| v.re * scale)
            .collect::<Array1<f64>>(),
        par_exp: Array1::from_vec((0..n_a).into_par_iter()
            .zip(a_fft_buf.par_iter())
            .map(|(_, v)| v.re * scale)
            .collect::<Vec<f64>>()))
}

/// Deconvolve two 1D signals using the Fast Fourier Transform (FFT).
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
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `Array1<f64>`: The FFT deconvolved result of the same length as input
///   signal `data_a`.
pub fn fft_deconvolve_1d<'a, T, A>(
    data_a: A,
    data_b: A,
    epsilon: Option<f64>,
    threads: Option<usize>,
) -> Array1<f64>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let data_a: ArrayBase<ViewRepr<&'a T>, Ix1> = data_a.into();
    let data_b: ArrayBase<ViewRepr<&'a T>, Ix1> = data_b.into();
    let epsilon = epsilon.unwrap_or(1e-8);
    let n_a = data_a.len();
    let n_b = data_b.len();
    let n_fft = n_a + n_b - 1;
    let fft_size = n_fft.next_power_of_two();
    let mut a_fft_buf = vec![Complex::zero(); fft_size];
    let mut b_fft_buf = vec![Complex::zero(); fft_size];
    let load_buffers = |a_buf: &mut Complex<f64>, b_buf: &mut Complex<f64>, a: &T, b: &T| {
        *a_buf = Complex::new(a.to_f64(), 0.0);
        *b_buf = Complex::new(b.to_f64(), 0.0);
    };
    par!(threads,
        seq_exp: Zip::from(&mut a_fft_buf[..n_a]).and(&mut b_fft_buf[..n_b])
            .and(data_a.view())
            .and(data_b.view())
            .for_each(&load_buffers),
        par_exp: Zip::from(&mut a_fft_buf[..n_a]).and(&mut b_fft_buf[..n_b])
            .and(data_a.view())
            .and(data_b.view())
            .par_for_each(&load_buffers));
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);
    fft.process(&mut a_fft_buf);
    fft.process(&mut b_fft_buf);
    // divide in the frequency domain with epsilon value and extract the real
    // component (scaled and input length trimmed)
    let div_calc = |a: &mut Complex<f64>, b: &Complex<f64>| {
        if a.norm_sqr() > epsilon {
            *a /= b;
        } else {
            *a = Complex::zero();
        }
    };
    par!(threads,
        seq_exp: a_fft_buf.iter_mut().zip(b_fft_buf.iter())
            .for_each(|(a, b)| div_calc(a, b)),
        par_exp: a_fft_buf.par_iter_mut().zip(b_fft_buf.par_iter())
            .for_each(|(a, b)| div_calc(a, b)));
    ifft.process(&mut a_fft_buf);
    let scale = 1.0 / fft_size as f64;
    par!(threads,
        seq_exp: (0..n_a).zip(a_fft_buf.iter()).map(|(_, v)| v.re * scale)
            .collect::<Array1<f64>>(),
        par_exp: Array1::from_vec((0..n_a).into_par_iter()
            .zip(a_fft_buf.par_iter())
            .map(|(_, v)| v.re * scale)
            .collect::<Vec<f64>>()))
}
