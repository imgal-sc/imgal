use ndarray::{Array1, ArrayBase, AsArray, Axis, Ix2, ViewRepr};
use rayon::prelude::*;

use crate::traits::numeric::AsNumeric;

/// TODO
pub fn graham_scan<'a, T, A>(data: A, parallel: bool) -> Array1<T>
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, Ix2> = data.into();
    // find the lowest point
    let init_idx: usize;
    if parallel {
        init_idx = data
            .axis_iter(Axis(0))
            .enumerate()
            .par_bridge()
            .min_by(|&(_, a), &(_, b)| {
                a[0].partial_cmp(&b[0])
                    .unwrap()
                    .then(a[1].partial_cmp(&b[1]).unwrap())
            })
            .unwrap()
            .0;
    } else {
        init_idx = data
            .axis_iter(Axis(0))
            .enumerate()
            .min_by(|&(_, a), &(_, b)| {
                a[0].partial_cmp(&b[0])
                    .unwrap()
                    .then(a[1].partial_cmp(&b[1]).unwrap())
            })
            .unwrap()
            .0;
    }
    dbg!(init_idx);
    todo!();
}
