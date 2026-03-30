use std::f32::consts::PI;

use ndarray::{Array, ArrayBase, ArrayViewMutD, AsArray, Dimension, ViewRepr, Zip};
use rayon::prelude::*;

use crate::constants::RNG_SEED;
use crate::simulation::rng::Pcg;
use crate::traits::numeric::AsNumeric;

/// TODO
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
                let l = a.to_f64() * scale;
                *b = get_poisson(&mut g, l as f32);
            });
    } else {
        Zip::from(data).and(noise_data.view_mut()).for_each(|a, b| {
            let l = a.to_f64() * scale;
            let x = get_poisson(&mut prng, l as f32);
            if x == T::default() {
                println!("{l}")
            }
            *b = x;
        });
    }
    noise_data
}

/// TODO
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
            let l = v.to_f64() * scale;
            *v = get_poisson(&mut g, l as f32);
        })
    } else {
        data.iter_mut().for_each(|v| {
            let l = v.to_f64() * scale;
            *v = get_poisson(&mut prng, l as f32);
        })
    }
}

/// TODO
fn get_poisson<T>(prng: &mut Pcg, lambda: f64) -> T
where
    T: AsNumeric,
{
    let thres = (-lambda as f32).exp();
    let mut prod: f32 = 1.0;
    let mut count: u64 = 0;
    loop {
        prod *= prng.next_f32();
        if prod < thres {
            return T::from_f64(count as f64);
        }
        count += 1;
    }
}
