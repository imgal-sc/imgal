use std::collections::HashSet;

use ndarray::{Array2, ArrayView3, Axis, Zip};

use crate::error::ImgalError;

/// Map G and S coordinates back to the input phasor array as a boolean mask.
///
/// # Description
///
/// This function maps the G and S coordinates back to the input G/S phasor
/// array and returns a 2-dimensional boolean mask where `true` indicates
/// G and S coordiantes presentin the `g_coords` and `s_coords` arrays.
///
/// # Arguments
///
/// * `data`: The G/S 3-dimensional array.
/// * `g_coords`: A 1-dimensional array of `g` coordinates in the `data` array.
///   The `g_coords` and `s_coords` array lengths must match.
/// * `s_coords`: A 1-dimensional array of `s` coordiantes in the `data` array.
///   The `s_coords` and `g_coords` array lengths must match.
/// * `axis`: The channel axis, default = 2.
///
/// # Returns
///
/// * `Ok(Array2<bool>)`: A 2-dimensional boolean mask where `true` pixels
///   represent values found in the `g_coords` and `s_coords` arrays.
/// * `Err(ImgalError)`: If "g" and "s" coordinate array lengths do not match.
pub fn gs_mask(
    data: ArrayView3<f64>,
    g_coords: &[f64],
    s_coords: &[f64],
    axis: Option<usize>,
) -> Result<Array2<bool>, ImgalError> {
    // check g and s coords array lengths
    let gl = g_coords.len();
    let sl = s_coords.len();
    if gl != sl {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_len: gl,
            b_arr_len: sl,
        });
    }

    // check if axis parameter is valid
    let a = axis.unwrap_or(2);
    if a >= 3 {
        return Err(ImgalError::InvalidAxis {
            axis_idx: a,
            dim_len: 3,
        });
    }

    // use a hash set of G/S coordinates for fast lookup
    let mut coords_set: HashSet<(u64, u64)> = HashSet::with_capacity(gl);
    g_coords.iter().zip(s_coords.iter()).for_each(|(g, s)| {
        coords_set.insert((g.to_bits(), s.to_bits()));
    });

    // create output array
    let mut shape = data.shape().to_vec();
    shape.remove(a);
    let mut map_arr = Array2::<bool>::default((shape[0], shape[1]));

    // check each pixel for matches in (g, s)
    let lanes = data.lanes(Axis(a));
    Zip::from(lanes)
        .and(map_arr.view_mut())
        .par_for_each(|ln, p| {
            let dg = ln[0];
            let ds = ln[1];
            if !dg.is_nan() || !ds.is_nan() || dg != 0.0 && ds != 0.0 {
                if coords_set.contains(&(dg.to_bits(), ds.to_bits())) {
                    *p = true;
                }
            }
        });

    // return output
    Ok(map_arr)
}

/// Compute the modulation of phasor G and S coordinates.
///
/// # Description
///
/// This function calculates the modulation (M) of phasor G and S coordinates
/// using the pythagorean theorem to find the hypotenuse (_i.e._ the
/// modulation):
///
/// ```text
/// M = √(G² + S²)
/// ````
///
/// # Arguments
///
/// * `g`: The real component, G.
/// * `s`: The imaginary component, S.
///
/// # Returns
///
/// * `f64`: The modulation (M) of the (G, S) phasor coordinates.
///
pub fn gs_modulation(g: f64, s: f64) -> f64 {
    let g_sqr: f64 = g * g;
    let s_sqr: f64 = s * s;
    f64::sqrt(g_sqr + s_sqr)
}

/// Compute the phase of phasor G and S coordinates.
///
/// # Description
///
/// This function calculates the phase or phi (φ) of phasor G and S coordinates
/// using:
///
/// ```text
/// φ = tan⁻¹(S / G)
/// ````
///
/// This implementation uses atan2 and computes the four quadrant arctangent of
/// the phasor coordinates.
///
/// # Arguments
///
/// * `g`: The real component, G.
/// * `s`: The imaginary component, S.
///
/// # Returns
///
/// * `f64`: The phase (phi, φ)  of the (G, S) phasor coordinates.
pub fn gs_phase(g: f64, s: f64) -> f64 {
    s.atan2(g)
}

/// Compute the G and S coordinates for a monoexponential decay.
///
/// # Description
///
/// This function computes the G and S coordinates for a monoexponential decay
/// given as:
///
/// ```text
/// G = 1 / 1 + (ωτ)²
/// S = ωτ / 1 + (ωτ)²
/// ```
///
/// # Arguments
///
/// * `tau`: The lifetime of a monoexponential decay.
/// * `omega`: The angular frequency.
///
/// # Returns
///
/// * `(f64, f64)`: The monoexponential decay coordinates, (G, S).
///
/// # Reference
///
/// <https://doi.org/10.1117/1.JBO.25.7.071203>
pub fn monoexponential_coordinates(tau: f64, omega: f64) -> (f64, f64) {
    let denom = 1.0 + (omega * tau).powi(2);
    let g = 1.0 / denom;
    let s = (omega * tau) / denom;
    (g, s)
}
