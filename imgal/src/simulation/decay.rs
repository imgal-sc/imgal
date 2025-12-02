use ndarray::{Array1, Array3, Zip};

use crate::error::ImgalError;
use crate::filter::fft_convolve_1d;
use crate::simulation::instrument;
use crate::statistics::sum;

/// Simulate a 1-dimensional Gaussian IRF convolved monoexponential or
/// multiexponential decay curve.
///
/// # Description
///
/// This function generates a 1-dimensonal Gaussian instrument response function
/// (IRF) convolved monoexponential or multiexponential decay curve. The ideal
/// decay curve is defined as the sum of one or more exponential components,
/// each characterized by a lifetime (tau) and fractional intensity:
///
/// ```text
/// I(t) = Σᵢ αᵢ × exp(-t/τᵢ)
/// ```
///
/// # Arguments
///
/// * `samples`: The number of discrete points that make up the decay curve.
/// * `period`: The period (_i.e._ time interval).
/// * `taus`: An array of lifetimes. For a monoexponential decay curve use a
///   single tau value and a fractional intensity of 1.0. For a multiexponential
///   decay curve use two or more tau values, matched with their respective
///   fractional intensity. The `taus` and `fractions` arrays must have the same
///   length. Tau values set to 0.0 will be skipped.
/// * `fractions`: An array of fractional intensities for each tau in the `taus`
///   array. The `fractions` array must be the same length as the `taus` array
///   and sum to 1.0. Fraction values set to 0.0 will be skipped.
/// * `total_counts`: The total intensity count (_e.g._ photon count) of the
///   decay curve.
/// * `irf_center`: The temporal position of the IRF peak within the time range.
/// * `irf_width`: The full width at half maximum (FWHM) of the IRF.
///
/// # Returns
///
/// * `Ok(Vec<f64>)`: The 1-dimensonal Gaussian IRF convolved monoexponential
///   or multiexponential decay curve.
/// * `Err(ImgalError)`: If taus and fractions array lengths do not match. If
///   fractions array does not sum to 1.0.
pub fn gaussian_exponential_decay_1d(
    samples: usize,
    period: f64,
    taus: &[f64],
    fractions: &[f64],
    total_counts: f64,
    irf_center: f64,
    irf_width: f64,
) -> Result<Vec<f64>, ImgalError> {
    let irf = instrument::gaussian_irf_1d(samples, period, irf_center, irf_width);
    let i_arr = ideal_exponential_decay_1d(samples, period, taus, fractions, total_counts)?;

    Ok(fft_convolve_1d(&i_arr, &irf))
}

/// Simulate a 3-dimensional Gaussian IRF convolved monoexponential or
/// multiexponential decay curve.
///
/// # Description
///
/// This function generates a 3-dimensonal Gaussian instrument response function
/// (IRF) convolved monoexponential or multiexponential decay curve. The ideal
/// decay curve is defined as the sum of one or more exponential components,
/// each characterized by a lifetime (tau) and fractional intensity:
///
/// ```text
/// I(t) = Σᵢ αᵢ × exp(-t/τᵢ)
/// ```
///
/// # Arguments
///
/// * `samples`: The number of discrete points that make up the decay curve.
/// * `period`: The period (_i.e._ time interval).
/// * `taus`: An array of lifetimes. For a monoexponential decay curve use a
///   single tau value and a fractional intensity of 1.0. For a
///   multiexponential decay curve use two or more tau values, matched with
///   their respective fractional intensity. The `taus` and `fractions` arrays
///   must have the same length. Tau values set to 0.0 will be skipped.
/// * `fractions`: An array of fractional intensities for each tau in the `taus`
///   array. The `fractions` array must be the same length as the `taus` array
///   and sum to 1.0. Fraction values set to 0.0 will be skipped.
/// * `total_counts`: The total intensity count (_e.g._ photon count) of the
///   decay curve.
/// * `irf_center`: The temporal position of the IRF peak within the time range.
/// * `irf_width`: The full width at half maximum (FWHM) of the IRF.
/// * `shape`: The row and col shape to broadcast the decay curve into.
///
/// # Returns
///
/// * `Ok(Array3<f64>)`: The 3-dimensional Gaussian IRF convolved
///   monoexponential or multiexponential decay curve.
/// * `Err(ImgalError)`: If taus and fractions array lengths do not match. If
///   fractions array does not sum to 1.0.
pub fn gaussian_exponential_decay_3d(
    samples: usize,
    period: f64,
    taus: &[f64],
    fractions: &[f64],
    total_counts: f64,
    irf_center: f64,
    irf_width: f64,
    shape: (usize, usize),
) -> Result<Array3<f64>, ImgalError> {
    // create 1-dimensional gaussian IRF convolved curve and broadcast
    let i_arr = gaussian_exponential_decay_1d(
        samples,
        period,
        taus,
        fractions,
        total_counts,
        irf_center,
        irf_width,
    )?;
    let i_arr = Array1::from_vec(i_arr);
    let dims = (shape.0, shape.1, samples);

    Ok(i_arr.broadcast(dims).unwrap().to_owned())
}

/// Simulate an ideal 1-dimensional monoexponential or multiexponential decay
/// curve.
///
/// # Description
///
/// This function generates a 1-dimensonal ideal exponential decay curve by
/// computing the sum of one or more exponential components, each characterized
/// by a lifetime (tau) and fractional intensity as defined by:
///
/// ```text
/// I(t) = Σᵢ αᵢ × exp(-t/τᵢ)
/// ```
///
/// where αᵢ are the pre-exponential factors derived from the fractional
/// intensities and lifetimes.
///
/// # Arguments
///
/// * `samples`: The number of discrete points that make up the decay curve.
/// * `period`: The period (_i.e._ time interval).
/// * `taus`: An array of lifetimes. For a monoexponential decay curve use a
///   single tau value and a fractional intensity of 1.0. For a
///   multiexponential decay curve use two or more tau values, matched with
///   their respective fractional intensity. The `taus` and `fractions` arrays
///   must have the same length. Tau values set to 0.0 will be skipped.
/// * `fractions`: An array of fractional intensities for each tau in the `taus`
///   array. The `fractions` array must be the same length as the `taus` array
///   and sum to 1.0. Fraction values set to 0.0 will be skipped.
/// * `total_counts`: The total intensity count (_e.g._ photon count) of the
///   decay curve.
///
/// # Returns
///
/// * `Ok(Vec<f64>)`: The 1-dimensonal monoexponential or multiexponential
///   decay curve.
/// * `Err(ImgalError)`: If taus and fractions array lengths do not match. If
///   fractions array does not sum to 1.0.
///
/// # Reference
///
/// <https://doi.org/10.1111/j.1749-6632.1969.tb56231.x>
pub fn ideal_exponential_decay_1d(
    samples: usize,
    period: f64,
    taus: &[f64],
    fractions: &[f64],
    total_counts: f64,
) -> Result<Vec<f64>, ImgalError> {
    // check taus and fractions array lengths
    let tl = taus.len();
    let fl = fractions.len();
    if tl != fl {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "taus",
            a_arr_len: tl,
            b_arr_name: "fractions",
            b_arr_len: fl,
        });
    }

    // create fractions array and check sum to 1.0
    let fs = sum(fractions);
    if fs != 1.0 {
        return Err(ImgalError::InvalidSum {
            expected: 1.0,
            got: fs,
        });
    }

    // create taus array and compute pre-exponential factors
    let frac_arr = Array1::from_vec(fractions.to_vec());
    let taus_arr = Array1::from_vec(taus.to_vec());
    let alph_arr = &frac_arr / &taus_arr;

    // create the time array and compute the intensity decay curve
    let mut i_arr = vec![0.0; samples];
    let time_arr = Array1::linspace(0.0, period, samples);
    alph_arr
        .iter()
        .zip(taus_arr.iter())
        .filter(|&(&al, &ta)| al != 0.0 && ta != 0.0)
        .for_each(|(al, ta)| {
            Zip::from(&mut i_arr).and(&time_arr).for_each(|i, t| {
                *i += al * (-t / ta).exp();
            });
        });

    // scale the histogram to total_counts
    let scale = total_counts / sum(&i_arr);
    i_arr.iter_mut().for_each(|v| *v *= scale);

    Ok(i_arr)
}

/// Simulate an ideal 3-dimensional monoexponential or multiexponential decay
/// curve.
///
/// # Description
///
/// This function generates a 3-dimensonal ideal exponential decay curve by
/// computing the sum of one or more exponential components, each characterized
/// by a lifetime (tau) and fractional intensity as defined by:
///
/// ```text
/// I(t) = Σᵢ αᵢ × exp(-t/τᵢ)
/// ```
///
/// where αᵢ are the pre-exponential factors derived from the fractional
/// intensities and lifetimes.
///
/// # Arguments
///
/// * `samples`: The number of discrete points that make up the decay curve.
/// * `period`: The period (_i.e._ time interval).
/// * `taus`: An array of lifetimes. For a monoexponential decay curve use a
///   single tau value and a fractional intensity of 1.0. For a
///   multiexponential decay curve use two or more tau values, matched with
///   their respective fractional intensity. The `taus` and `fractions` arrays
///   must have the same length. Tau values set to 0.0 will be skipped.
/// * `fractions`: An array of fractional intensities for each tau in the `taus`
///   array. The `fractions` array must be the same length as the `taus` array
///   and sum to 1.0. Fraction values set to 0.0 will be skipped.
/// * `total_counts`: The total intensity count (_e.g._ photon count) of the
///   decay curve.
/// * `shape`: The row and col shape to broadcast the decay curve into.
///
/// # Returns
///
/// * `Ok(Array3<f64>)`: The 3-dimensonal monoexponential or multiexponential
///   decay curve.
/// * `Err(ImgalError)`: If taus and fractions array lengths do not match. If
///   fractions array does not sum to 1.0.
///
/// # Reference
///
/// <https://doi.org/10.1111/j.1749-6632.1969.tb56231.x>
pub fn ideal_exponential_decay_3d(
    samples: usize,
    period: f64,
    taus: &[f64],
    fractions: &[f64],
    total_counts: f64,
    shape: (usize, usize),
) -> Result<Array3<f64>, ImgalError> {
    // create 1-dimensional decay curve and broadcast
    let i_arr = ideal_exponential_decay_1d(samples, period, taus, fractions, total_counts)?;
    let i_arr = Array1::from_vec(i_arr);
    let dims = (shape.0, shape.1, samples);

    Ok(i_arr.broadcast(dims).unwrap().to_owned())
}

/// Simulate a 1-dimensional IRF convolved monoexponential or multiexponential
/// decay curve.
///
/// # Description
///
/// This function generates a 1-dimensonal instrument response function (IRF)
/// convolved monoexponential or multiexponential decay curve. The ideal
/// decay curve is defined as the sum of one or more exponential components,
/// each characterized by a lifetime (tau) and fractional intensity:
///
/// ```text
/// I(t) = Σᵢ αᵢ × exp(-t/τᵢ)
/// ```
///
/// # Arguments
///
/// * `irf`: The IRF as a 1-dimensonal array.
/// * `samples`: The number of discrete points that make up the decay curve.
/// * `period`: The period (_i.e._ time interval).
/// * `taus`: An array of lifetimes. For a monoexponential decay curve use a
///   single tau value and a fractional intensity of 1.0. For a
///   multiexponential decay curve use two or more tau values, matched with
///   their respective fractional intensity. The `taus` and `fractions` arrays
///   must have the same length. Tau values set to 0.0 will be skipped.
/// * `fractions`: An array of fractional intensities for each tau in the `taus`
///   array. The `fractions` array must be the same length as the `taus` array
///   and sum to 1.0. Fraction values set to 0.0 will be skipped.
/// * `total_counts`: The total intensity count (_e.g._ photon count) of the
///   decay curve.
///
/// # Returns
///
/// * `Ok(Vec<f64>)`: The 1-dimensional IRF convolved monoexponential or
///   multiexponential decay curve.
/// * `Err(ImgalError)`: If taus and fractions array lengths do not match. If
///   fractions array does not sum to 1.0.
pub fn irf_exponential_decay_1d(
    irf: &[f64],
    samples: usize,
    period: f64,
    taus: &[f64],
    fractions: &[f64],
    total_counts: f64,
) -> Result<Vec<f64>, ImgalError> {
    // create ideal decay curve and convolve with input irf
    let i_arr = ideal_exponential_decay_1d(samples, period, taus, fractions, total_counts)?;

    Ok(fft_convolve_1d(i_arr.as_slice(), irf))
}

/// Simulate a 3-dimensional IRF convolved monoexponential or multiexponential
/// decay curve.
///
/// # Description
///
/// This function generates a 3-dimensonal instrument response function (IRF)
/// convolved monoexponential or multiexponential decay curve. The ideal
/// decay curve is defined as the sum of one or more exponential components,
/// each characterized by a lifetime (tau) and fractional intensity:
///
/// ```text
/// I(t) = Σᵢ αᵢ × exp(-t/τᵢ)
/// ```
///
/// # Arguments
///
/// * `irf`: The IRF as a 1-dimensonal array.
/// * `samples`: The number of discrete points that make up the decay curve.
/// * `period`: The period (_i.e._ time interval).
/// * `taus`: An array of lifetimes. For a monoexponential decay curve use a
///   single tau value and a fractional intensity of 1.0. For a
///   multiexponential decay curve use two or more tau values, matched with
///   their respective fractional intensity. The `taus` and `fractions` arrays
///   must have the same length. Tau values set to 0.0 will be skipped.
/// * `fractions`: An array of fractional intensities for each tau in the `taus`
///   array. The `fractions` array must be the same length as the `taus` array
///   and sum to 1.0. Fraction values set to 0.0 will be skipped.
/// * `total_counts`: The total intensity count (_e.g._ photon count) of the
///   decay curve.
/// * `shape`: The row and col shape to broadcast the decay curve into.
///
/// # Returns
///
/// * `Ok(Array3<f64>)`: The 3-dimensional IRF convolved monoexponential or
///   multiexponential decay curve.
/// * `Err(ImgalError)`: If taus and fractions array lengths do not match. If
///   fractions array does not sum to 1.0.
pub fn irf_exponential_decay_3d(
    irf: &[f64],
    samples: usize,
    period: f64,
    taus: &[f64],
    fractions: &[f64],
    total_counts: f64,
    shape: (usize, usize),
) -> Result<Array3<f64>, ImgalError> {
    // create 1-dimensional IRF convolved decay curve to broadcast
    let i_arr = irf_exponential_decay_1d(irf, samples, period, taus, fractions, total_counts)?;
    let i_arr = Array1::from_vec(i_arr);
    let dims = (shape.0, shape.1, samples);

    Ok(i_arr.broadcast(dims).unwrap().to_owned())
}
