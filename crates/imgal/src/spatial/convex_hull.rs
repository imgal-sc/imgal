use std::cmp::Ordering;

use ndarray::{Array1, ArrayBase, AsArray, Axis, Ix2, ViewRepr};
use rayon::prelude::*;

use crate::traits::numeric::AsNumeric;

/// TODO
pub fn graham_scan<'a, T, A>(points: A, parallel: bool) -> Array1<T>
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    let points: ArrayBase<ViewRepr<&'a T>, Ix2> = points.into();
    let axis = Axis(0);
    // start the Graham scan by finding the lowest (row) and most left (col)
    // point
    let pivot_idx: usize;
    if parallel {
        pivot_idx = points
            .axis_iter(axis)
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
        pivot_idx = points
            .axis_iter(axis)
            .enumerate()
            .min_by(|&(_, a), &(_, b)| {
                a[0].partial_cmp(&b[0])
                    .unwrap()
                    .then(a[1].partial_cmp(&b[1]).unwrap())
            })
            .unwrap()
            .0;
    }
    let pivot_pos = (points[[pivot_idx, 0]], points[[pivot_idx, 1]]);
    let mut angle_sorted_pos: Vec<(T, T)> = points
        .axis_iter(axis)
        .enumerate()
        .map(|(_, p)| (p[0], p[1]))
        .collect();
    angle_sorted_pos.swap(0, pivot_idx);
    angle_sorted_pos[1..].sort_by(|&a, &b| {
        let cross = cross_prod_2d(pivot_pos, a, b).to_f64();
        if cross.abs() < 1e-12 {
            // points a and b are collinear
            dist_sq_2d(pivot_pos, a)
                .partial_cmp(&dist_sq_2d(pivot_pos, b))
                .unwrap()
        } else if cross > 0.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    });
    dbg!(angle_sorted_pos);
    todo!();
}

/// TODO here its (row, col)
fn cross_prod_2d<T>(pivot: (T, T), point_a: (T, T), point_b: (T, T)) -> T
where
    T: AsNumeric,
{
    (point_a.1 - pivot.1) * (point_b.0 - pivot.0) - (point_a.0 - pivot.0) * (point_b.1 - pivot.1)
}

/// TODO
fn dist_sq_2d<T>(point_a: (T, T), point_b: (T, T)) -> T
where
    T: AsNumeric,
{
    let dy = point_a.0 - point_b.0;
    let dx = point_a.1 - point_b.1;
    dx * dx + dy * dy
}
