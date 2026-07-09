//! SIMD compiler hints
//!
//! This module provides functions that help hint to compiler to autovectorize
//! loops (*i.e.* unrolling) and utilize SIMD.

use ndarray::{ArrayView, Dimension};

use crate::prelude::*;

/// Fold over an n-dimensional array view with autovectorization hints.
///
/// # Description
///
/// todo
///
/// # Arguments
///
/// todo
///
/// # Returns
///
/// todo
#[inline(always)]
pub fn fast_fold<T, D, B, F>(data: ArrayView<T, D>, init: B, f: F) -> T
where
    B: Fn() -> T + Copy,
    D: Dimension,
    F: Fn(T, T) -> T + Copy,
    T: AsNumeric,
{
    if let Some(s) = data.as_slice_memory_order() {
        unrolled_fold(s, init, f)
    } else {
        data.rows().into_iter().fold(init(), |acc, r| {
            if let Some(s) = r.as_slice_memory_order() {
                let res = unrolled_fold(s, init, f);
                f(acc, res)
            } else {
                let res = r.iter().fold(init(), |acc, &v| f(acc, v));
                f(acc, res)
            }
        })
    }
}

/// Fold over a slice using an eight-chain unrolled reduction.
///
/// # Description
///
/// Computes a reduction of `data` using a provided initialization (`init`) and
/// binary operation closure (`f`) using 8 indpendent accumulation chains to
/// encourage the compiler to use autovectorization. This function is adapted
/// from ndarray's numeric_utils module, written by bluss.
///
/// # Arguments
///
/// * `data`: The input slice to fold.
/// * `init`: An initialization closure (*e.g* `T::default` or `|| 0.0`).
/// * `f`: A binary operation closure (*e.g* `T::add` or `|a, b| a + b `).
///
/// # Returns
///
/// * `T`: The reduced value across all elements in `data`.
pub fn unrolled_fold<T, B, F>(data: &[T], init: B, f: F) -> T
where
    B: Fn() -> T,
    F: Fn(T, T) -> T,
    T: AsNumeric,
{
    let mut acc = init();
    let mut chains: (T, T, T, T, T, T, T, T) = (
        init(),
        init(),
        init(),
        init(),
        init(),
        init(),
        init(),
        init(),
    );
    let (chunks, remainder) = data.as_chunks::<8>();
    chunks.iter().for_each(|c| {
        chains.0 = f(chains.0, c[0]);
        chains.1 = f(chains.1, c[1]);
        chains.2 = f(chains.2, c[2]);
        chains.3 = f(chains.3, c[3]);
        chains.4 = f(chains.4, c[4]);
        chains.5 = f(chains.5, c[5]);
        chains.6 = f(chains.6, c[6]);
        chains.7 = f(chains.7, c[7]);
    });
    acc = f(acc, f(chains.0, chains.4));
    acc = f(acc, f(chains.1, chains.5));
    acc = f(acc, f(chains.2, chains.6));
    acc = f(acc, f(chains.3, chains.7));
    remainder.iter().for_each(|&v| acc = f(acc, v));
    acc
}
