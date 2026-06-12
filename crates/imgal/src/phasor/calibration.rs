use ndarray::{
    Array3, ArrayBase, ArrayView1, ArrayViewMut1, ArrayViewMut3, AsArray, Axis, Ix3, ViewRepr, Zip,
};
use rayon::prelude::*;

use crate::phasor::plot;
use crate::prelude::*;

/// Calibrate a real and imaginary (G, S) coordinates.
///
/// # Description
///
/// Calibrates the real and imaginary (*e.g.* G and S) coordinates by rotating
/// and scaling by phase (φ) and modulation (M) respectively using:
///
/// ```text
/// g = M * cos(φ)
/// s = M * sin(φ)
/// G' = G * g - S * s
/// S' = G * s + S * g
/// ```
///
/// Where G' and S' are the calibrated real and imaginary values after rotation
/// and scaling.
///
/// # Arguments
///
/// * `g`: The real component (G) to calibrate.
/// * `s`: The imaginary (S) to calibrate.
/// * `modulation`: The modulation to scale the input (G, S) coordinates.
/// * `phase`: The phase, φ angle, to rotate the input (G, S) coordinates.
///
/// # Returns
///
/// * `(f64, f64)`: The calibrated coordinates, (G, S).
#[inline]
pub fn calibrate_coords(g: f64, s: f64, modulation: f64, phase: f64) -> (f64, f64) {
    let g_trans = modulation * phase.cos();
    let s_trans = modulation * phase.sin();
    let g_cal = g * g_trans - s * s_trans;
    let s_cal = g * s_trans + s * g_trans;
    (g_cal, s_cal)
}

/// Calibrate a real and imaginary (G, S) 3D phasor image.
///
/// # Description
///
/// Calibrates an input 3D phasor image by rotating and scaling G and S
/// coordinates by phase (φ) and modulation (M) respectively using:
///
/// ```text
/// g = M * cos(φ)
/// s = M * sin(φ)
/// G' = G * g - S * s
/// S' = G * s + S * g
/// ```
///
/// Where G' and S' are the calibrated real and imaginary values after rotation
/// and scaling.
///
/// # Arguments
///
/// * `data`: The input 3D phasor image, where G and S are channels `0` and `1`
///   respectively.
/// * `modulation`: The modulation to scale the input (G, S) coordinates.
/// * `phase`: The phase, φ angle, to rotate the input (G, S) coordinates.
/// * `axis`: The channel axis. If `None`, then `axis = 2`.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `Array3<f64>`: A 3D image with the calibrated phasor values, where
///   calibrated G and S are channels `0` and `1` respectively.
#[inline]
pub fn calibrate_gs_image<'a, T, A>(
    data: A,
    modulation: f64,
    phase: f64,
    axis: Option<usize>,
    threads: Option<usize>,
) -> Array3<f64>
where
    A: AsArray<'a, T, Ix3>,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, Ix3> = data.into();
    let axis = axis.unwrap_or(2);
    let shape = data.dim();
    let mut c_data = Array3::<f64>::zeros(shape);
    let g_trans = modulation * phase.cos();
    let s_trans = modulation * phase.sin();
    let src_lanes = data.lanes(Axis(axis));
    let dst_lanes = c_data.lanes_mut(Axis(axis));
    let gs_calibration_calc = |s: ArrayView1<T>, d: &mut ArrayViewMut1<f64>| {
        d[0] = s[0].to_f64() * g_trans - s[1].to_f64() * s_trans;
        d[1] = s[0].to_f64() * s_trans + s[1].to_f64() * g_trans;
    };
    par!(threads,
        seq_exp: Zip::from(src_lanes).and(dst_lanes)
            .for_each(|s, mut d| gs_calibration_calc(s, &mut d)),
        par_exp: Zip::from(src_lanes).and(dst_lanes)
            .par_for_each(|s, mut d| gs_calibration_calc(s, &mut d)));
    c_data
}

/// Calibrate a real and imaginary (G, S) 3D phasor image.
///
/// # Description
///
/// Calibrates an input 3D phasor image by rotating and scaling G and S
/// coordinates by phase (φ) and modulation (M) respectively using:
///
/// ```text
/// g = M * cos(φ)
/// s = M * sin(φ)
/// G' = G * g - S * s
/// S' = G * s + S * g
/// ```
///
/// Where G' and S' are the calibrated real and imaginary values after rotation
/// and scaling.
///
/// # Arguments
///
/// * `data`: The input 3D phasor image, where G and S are channels `0` and `1`
///   respectively.
/// * `modulation`: The modulation to scale the input (G, S) coordinates.
/// * `phase`: The phase, φ angle, to rotate the input (G, S) coordinates.
/// * `axis`: The channel axis. If `None`, then `axis = 2`.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
#[inline]
pub fn calibrate_gs_image_mut(
    mut data: ArrayViewMut3<f64>,
    modulation: f64,
    phase: f64,
    axis: Option<usize>,
    threads: Option<usize>,
) {
    let axis = axis.unwrap_or(2);
    let g_trans = modulation * phase.cos();
    let s_trans = modulation * phase.sin();
    let lanes = data.lanes_mut(Axis(axis));
    let gs_calibration_calc = |ln: &mut ArrayViewMut1<f64>| {
        let g_cal = ln[0] * g_trans - ln[1] * s_trans;
        let s_cal = ln[0] * s_trans + ln[1] * g_trans;
        ln[0] = g_cal;
        ln[1] = s_cal;
    };
    par!(threads,
        seq_exp: lanes.into_iter().for_each(|mut ln| gs_calibration_calc(&mut ln)),
        par_exp: lanes.into_iter().par_bridge()
            .for_each(|mut ln| gs_calibration_calc(&mut ln)))
}

/// Compute the modulation and phase calibration values.
///
/// # Description
///
/// Computes the modulation and phase calibration values from theoretical
/// monoexponential coordinates (computed from `tau` and `omega`) and measured
/// coordinates. The output, (M, φ), are the modulation and phase values to
/// calibrate with.
///
/// # Arguments
///
/// * `g`: The measured real (G) value.
/// * `s`: The measured imaginary (S) value.
/// * `tau`: The lifetime, τ.
/// * `omega`: The angular frequency, ω.
///
/// # Returns
///
/// * `(f64, f64)`: The modulation and phase calibration values, (M, φ).
#[inline]
pub fn modulation_and_phase(g: f64, s: f64, tau: f64, omega: f64) -> (f64, f64) {
    // compute the reference monoexponential modulation and phase then return
    // the difference between the measured G/S modulation and phase
    let cal_point = plot::monoexponential_coords(tau, omega);
    let cal_mod = plot::gs_modulation(cal_point.0, cal_point.1);
    let cal_phs = plot::gs_phase(cal_point.0, cal_point.1);
    let data_mod = plot::gs_modulation(g, s);
    let data_phs = plot::gs_phase(g, s);
    let d_mod = cal_mod / data_mod;
    let d_phs = cal_phs - data_phs;
    (d_mod, d_phs)
}
