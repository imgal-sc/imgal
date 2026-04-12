use std::cmp::Ordering;

use ndarray::{Array2, ArrayBase, AsArray, Axis, Ix2, ViewRepr};
use rayon::prelude::*;

use crate::traits::numeric::AsNumeric;

/// TODO
pub fn graham_scan<'a, T, A>(points: A, parallel: bool) -> Array2<T>
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    let points: ArrayBase<ViewRepr<&'a T>, Ix2> = points.into();
    let axis = Axis(0);
    let n = points.dim().0;
    // start by finding the lowest (row) and most left (col) point
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
    // sort the rest of the lowest points by polar angle relative to the pivot
    // point to set the order the points are visited in the scan
    let pivot_pos = (points[[pivot_idx, 0]], points[[pivot_idx, 1]]);
    let mut point_inds: Vec<usize> = (0..n).map(|i| i).collect();
    point_inds.swap(0, pivot_idx);
    point_inds[1..].sort_by(|&a, &b| {
        let a_pos = (points[[a, 0]], points[[a, 1]]);
        let b_pos = (points[[b, 0]], points[[b, 1]]);
        let cross = cross_prod_2d(pivot_pos, a_pos, b_pos).to_f64();
        if cross.abs() < 1e-12 {
            // points a and b are collilnear
            dist_sq_2d(pivot_pos, a_pos)
                .partial_cmp(&dist_sq_2d(pivot_pos, b_pos))
                .unwrap()
        } else if cross > 0.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    });
    let hull = point_inds.iter().fold(Vec::with_capacity(n), |mut hull, &i| {
        let cur_pos = (points[[i, 0]], points[[i, 1]]);
        while hull.len() >= 2 {
            let top = hull[hull.len() - 1];
            let second = hull[hull.len() - 2];
            if cross_prod_2d(second, top, cur_pos).to_f64() <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(cur_pos);
        hull
    });
    Array2::from_shape_vec((hull.len(), 2), hull.iter().flat_map(|&(r, c)| [r, c]).collect()).unwrap()
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
