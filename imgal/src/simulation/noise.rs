use ndarray::{
    Array3, ArrayBase, ArrayView1, ArrayViewMut3, AsArray, Axis, Ix1, Ix3, ViewRepr, Zip,
};
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson};
use rayon::prelude::*;

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Apply Poisson noise on a 1-dimensional array.
///
/// # Description
///
/// Applies Poisson noise (_i.e._ shot noise) on a 1-dimensional array of data.
/// An element-wise lambda value (scaled by the `scale` parameter) is used to
/// simulate the Poisson noise with variable signal strength.
///
/// This function creates a new array and does not mutate the input array.
///
/// # Arguments
///
/// * `data`: The input 1-dimensional array.
/// * `scale`: The scale factor.
/// * `seed`: Pseudorandom number generator seed. Set the `seed` value to apply
///   homogenous noise to the input array. If `None`, then heterogenous noise
///   is applied to the input array.
///
/// # Returns
///
/// * `Vec<f64>`: A 1-dimensional array of the input data with Poisson noise
///   applied.
pub fn poisson_noise_1d<'a, T, A>(data: A, scale: f64, seed: Option<u64>) -> Vec<f64>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let view: ArrayBase<ViewRepr<&'a T>, Ix1> = data.into();
    let s = seed.unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(s);
    let mut n_data = vec![0.0; view.len()];
    n_data.iter_mut().zip(view.iter()).for_each(|(n, &d)| {
        if d.to_f64() > 0.0 {
            let l: f64 = d.to_f64() * scale;
            let p = Poisson::new(l).unwrap();
            *n = p.sample(&mut rng);
        } else {
            *n = 0.0;
        }
    });

    n_data
}

/// Apply Poisson noise on a 1-dimensional array.
///
/// # Description
///
/// Applies Poisson noise (_i.e._ shot noise) on a 1-dimensional array of data.
/// An element-wise lambda value (scaled by the `scale` parameter) is used to
/// simulate the Poisson noise with variable signal strength.
///
/// This function mutates the input array and does not create a new array.
///
/// # Arguments
///
/// * `data`: The input 1-dimensional array view to mutate.
/// * `scale`: The scale factor.
/// * `seed`: Pseudorandom number generator seed. Set the `seed` value to apply
///   homogenous noise to the input array. If `None`, then heterogenous noise
///   is applied to the input array.
pub fn poisson_noise_1d_mut(data: &mut [f64], scale: f64, seed: Option<u64>) {
    let s = seed.unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(s);
    data.iter_mut().for_each(|x| {
        if *x > 0.0 {
            let l = *x * scale;
            let p = Poisson::new(l).unwrap();
            *x = p.sample(&mut rng);
        } else {
            *x = 0.0;
        }
    });
}

/// Apply Poisson noise on a 3-dimensional array.
///
/// # Description
///
/// Applies Poisson noise (_i.e._ shot noise) on a 3-dimensional array of data.
/// An element-wise lambda value (scaled by the `scale` parameter) is used to
/// simulate Poisson noise with variable signal strength.
///
/// This function creates a new array and does not mutate the input array.
///
/// # Arguments
///
/// * `data`: The input 3-dimensional array.
/// * `scale`: The scale factor.
/// * `seed`: Pseudorandom number generator seed. Set the `seed` value to apply
///   homogenous noise to the input array. If `None`, then heterogenous noise
///   is applied to the input array.
/// * `axis`: The signal data axis. If `None`, then `axis = 2`.
///
/// # Returns
///
/// * `Ok(Array3<f64>)`: A 3-dimensional array of the input data with Poisson noise
///   applied.
/// * `Err(ImgalError)`: If `axis >= 3`.
pub fn poisson_noise_3d<'a, T, A>(
    data: A,
    scale: f64,
    seed: Option<u64>,
    axis: Option<usize>,
) -> Result<Array3<f64>, ImgalError>
where
    A: AsArray<'a, T, Ix3>,
    T: 'a + AsNumeric,
{
    // validate the axis value
    let a = axis.unwrap_or(2);
    if a >= 3 {
        return Err(ImgalError::InvalidAxis {
            axis_idx: a,
            dim_len: 3,
        });
    }

    // apply homogenous Poisson noise (with seed value set) or heterogenous
    // noise (no seed value set, each lane gets a new rng)
    let view: ArrayBase<ViewRepr<&'a T>, Ix3> = data.into();
    let shape = view.dim();
    let mut n_data = Array3::<f64>::zeros(shape);
    let src_lanes = view.lanes(Axis(a));
    let dst_lanes = n_data.lanes_mut(Axis(a));
    if let Some(s) = seed {
        Zip::from(src_lanes)
            .and(dst_lanes)
            .par_for_each(|s_ln, d_ln| {
                let mut rng = StdRng::seed_from_u64(s);
                Zip::from(s_ln).and(d_ln).for_each(|s, d| {
                    if (*s).to_f64() > 0.0 {
                        let l = (*s).to_f64() * scale;
                        let p = Poisson::new(l).unwrap();
                        *d = p.sample(&mut rng);
                    } else {
                        *d = 0.0;
                    }
                });
            });
    } else {
        Zip::from(src_lanes)
            .and(dst_lanes)
            .par_for_each(|s_ln, d_ln| {
                let mut rng = rand::rng();
                Zip::from(s_ln).and(d_ln).for_each(|s, d| {
                    if (*s).to_f64() > 0.0 {
                        let l = (*s).to_f64() * scale;
                        let p = Poisson::new(l).unwrap();
                        *d = p.sample(&mut rng);
                    } else {
                        *d = 0.0
                    }
                });
            });
    }

    Ok(n_data)
}

/// Apply Poisson noise on a 3-dimensional array.
///
/// # Description
///
/// Applies Poisson noise (_i.e._ shot noise) on a 3-dimensional array of data.
/// An element-wise lambda value (scaled by the `scale` parameter) is used to
/// simulate Poisson noise with variable signal strength.
///
/// This function mutates the input array and does not create a new array.
///
/// # Arguments
///
/// * `data`: The input 3-dimensional array to mutate.
/// * `scale`: The scale factor.
/// * `seed`: Pseudorandom number generator seed. Set the `seed` value to apply
///   homogenous noise to the input array. If `None`, then heterogenous noise
///   is applied to the input array.
/// * `axis`: The signal data axis. If `None`, then `axis = 2`.
pub fn poisson_noise_3d_mut(
    mut data: ArrayViewMut3<f64>,
    scale: f64,
    seed: Option<u64>,
    axis: Option<usize>,
) {
    // apply homogenous Poisson noise (with seed value set) or heterogenous
    // noise (no seed value set, each lane gets a new rng)
    let a = axis.unwrap_or(2);
    let lanes = data.lanes_mut(Axis(a));
    if let Some(s) = seed {
        lanes.into_iter().par_bridge().for_each(|mut ln| {
            if let Some(l) = ln.as_slice_mut() {
                poisson_noise_1d_mut(l, scale, Some(s));
            } else {
                let mut l = ln.to_vec();
                poisson_noise_1d_mut(&mut l, scale, Some(s));
                let l = ArrayView1::from(&l);
                ln.assign(&l);
            }
        });
    } else {
        lanes.into_iter().par_bridge().for_each(|mut ln| {
            let mut rng = rand::rng();
            let s = rng.next_u64();
            if let Some(l) = ln.as_slice_mut() {
                poisson_noise_1d_mut(l, scale, Some(s));
            } else {
                let mut l = ln.to_vec();
                poisson_noise_1d_mut(&mut l, scale, Some(s));
                let l = ArrayView1::from(&l);
                ln.assign(&l);
            }
        });
    }
}
