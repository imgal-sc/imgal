use std::f32::consts::PI;

use ndarray::{Array, ArrayBase, ArrayViewMutD, AsArray, Dimension, ViewRepr, Zip};
use rayon::prelude::*;

use crate::constants::RNG_SEED;
use crate::simulation::rng::Pcg;
use crate::traits::numeric::AsNumeric;

/// Create a new n-dimensional image with Poisson noise.
///
/// # Description
///
/// Creates a new n-dimensional image of the input data with scaled Poisson
/// noise (*i.e.* shot noise) using Knuth's algorithm.
///
/// # Arguments
///
/// * `data`: The input n-dimensonal image.
/// * `scale`: The noise scale factor. Smaller values produce noiser output,
///   while larger values produce output closer to the original input.
/// * `seed`: The seed value for the pseudo-random number generator.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Array<T, D>`: An image of the same dimensions as the input `data`, where
///   each element is a Poisson-distributed sample derived from the
///   corresponding input value.i
///
/// # Reference
///
/// <https://en.wikipedia.org/wiki/Poisson_distribution>
pub fn poisson_noise<'a, T, A, D>(
    data: A,
    scale: f64,
    seed: Option<u64>,
    parallel: bool,
) -> Array<T, D>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let seed = seed.unwrap_or(RNG_SEED);
    let mut prng = Pcg::new(seed);
    let mut noise_data: Array<T, D> = Array::from_elem(data.dim(), T::default());
    if parallel {
        Zip::from(data)
            .and(noise_data.view_mut())
            .into_par_iter()
            .for_each_with(prng.fork(), |mut g, (a, b)| {
                let a = a.to_f64();
                let s = if a < 0.0 { -1.0 } else { 1.0 };
                let l = a.abs() * scale;
                *b = T::from_f64(get_poisson(&mut g, l as f32) * s);
            });
    } else {
        Zip::from(data).and(noise_data.view_mut()).for_each(|a, b| {
            let a = a.to_f64();
            let s = if a < 0.0 { -1.0 } else { 1.0 };
            let l = a.abs() * scale;
            *b = T::from_f64(get_poisson(&mut prng, l as f32) * s);
        });
    }
    noise_data
}

/// Mutate an n-dimensional image with Poisson noise.
///
/// # Description
///
/// Mutates an n-dimensional image with scaled Poisson noise (*i.e.* shot noise)
/// using Knuth's algorithm.
///
/// # Arguments
///
/// * `data`: The input n-dimensonal image to mutate.
/// * `scale`: The noise scale factor. Smaller values produce noiser output,
///   while larger values produce output closer to the original input.
/// * `seed`: The seed value for the pseudo-random number generator.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Reference
///
/// <https://en.wikipedia.org/wiki/Poisson_distribution>
pub fn poisson_noise_mut<T>(
    mut data: ArrayViewMutD<T>,
    scale: f64,
    seed: Option<u64>,
    parallel: bool,
) where
    T: AsNumeric,
{
    let seed = seed.unwrap_or(RNG_SEED);
    let mut prng = Pcg::new(seed);
    if parallel {
        data.into_par_iter().for_each_with(prng.fork(), |mut g, v| {
            let a = v.to_f64();
            let s = if a < 0.0 { -1.0 } else { 1.0 };
            let l = a.abs() * scale;
            *v = T::from_f64(get_poisson(&mut g, l as f32) * s);
        })
    } else {
        data.iter_mut().for_each(|v| {
            let a = v.to_f64();
            let s = if a < 0.0 { -1.0 } else { 1.0 };
            let l = a.to_f64() * scale;
            *v = T::from_f64(get_poisson(&mut prng, l as f32) * s);
        })
    }
}

/// Get the a Poisson value.
///
/// # Description
///
/// This function generates random Poisson distributed numbers using Knuth's
/// algorithm. When lambda values are larger than `30.0`, the Box-Muller
/// transform fallback is used.
///
/// # Arguments
///
/// * `prng`: An instances of a PCG pseudo-random number generator.
/// * `lambda`: The lambda value.
///
/// # Returns
///
/// * `f64`: The Poisson value.
///
/// # Reference
///
/// <https://en.wikipedia.org/wiki/Poisson_distribution>
/// <https://en.wikipedia.org/wiki/Box-Muller_transform>
fn get_poisson(prng: &mut Pcg, lambda: f32) -> f64 {
    // use the basic form of the Box-Muller transform for normal approximation
    // if lambda is too large (it overflows and prod can never be smaller) for
    // Knuth's algorithm
    if lambda >= 30.0 {
        let u1 = prng.next_f32();
        let u2 = prng.next_f32();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        let sample = (lambda + lambda.sqrt() * z).round().max(0.0);
        return sample as f64;
    }
    let thres = (-lambda).exp();
    let mut prod: f32 = 1.0;
    let mut count: u64 = 0;
    loop {
        prod *= prng.next_f32();
        if prod < thres {
            return count as f64;
        }
        count += 1;
    }
}
