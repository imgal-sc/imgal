use std::cmp::Ordering;

use ndarray::{Array2, ArrayBase, AsArray, Axis, Ix2, ViewRepr};
use rayon::prelude::*;

use crate::traits::numeric::AsNumeric;

/// Create a convex hull from a 2D point cloud using the Graham scan method.
///
/// # Description
///
/// Constructs a 2D convex hull from a 2D point cloud using the Graham scan
/// method, where points are sorted by their polar angle relative to the pivot
/// point (the lowest and most left point). The convex hull is constructed by
/// processing these angle sorted points and retaining only those where each
/// point makes a left turn relative to the last two hull vertices.
///
/// # Arguments
///
/// * `points`: The 2D point cloud with shape `(n_points, 2)`.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Array2<T>`: The points that comprise the convex hull.
///
/// # Reference
///
/// <https://doi.org/10.1016/0020-0190(72)90045-2>
/// <https://en.wikipedia.org/wiki/Graham_scan>
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
            // points a and b are collinear
            dist_sq_2d(pivot_pos, a_pos)
                .partial_cmp(&dist_sq_2d(pivot_pos, b_pos))
                .unwrap()
        } else if cross > 0.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    });
    let hull = point_inds
        .iter()
        .fold(Vec::with_capacity(n), |mut hull, &i| {
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
    Array2::from_shape_vec(
        (hull.len(), 2),
        hull.iter().flat_map(|&(r, c)| [r, c]).collect(),
    )
    .unwrap()
}

/// Compute the 2D cross product of vectors defined by three points.
///
/// # Description
///
/// Calculates the cross product of vectors `(point_a - pivot)` and
/// `(point_b - pivot)`. The result indicates rotational direction:
///
/// Positive => counter-clockwise (left) turn
/// Negative => clock-wise (right) turn
/// Zero => collinear points
///
/// # Arguments
///
/// * `pivot`: The pivot point as (row, col).
/// * `point_a`: The first point as (row, col).
/// * `point_b`: The second point as (row, col).
///
/// # Returns
///
/// * `T`: The cross product.
fn cross_prod_2d<T>(pivot: (T, T), point_a: (T, T), point_b: (T, T)) -> T
where
    T: AsNumeric,
{
    (point_a.1 - pivot.1) * (point_b.0 - pivot.0) - (point_a.0 - pivot.0) * (point_b.1 - pivot.1)
}

/// Comppute the squared Euclidean distance between two points.
///
/// # Description
///
/// Calculates the squared distance between two 2D points.
///
/// # Arguments
///
/// * `point_a`: The first point as (row, col).
/// * `point_b`: The second point as (row, col).
///
/// # Returns
///
/// * `T`: The squared Euclidean distance.
fn dist_sq_2d<T>(point_a: (T, T), point_b: (T, T)) -> T
where
    T: AsNumeric,
{
    let dy = point_a.0 - point_b.0;
    let dx = point_a.1 - point_b.1;
    dx * dx + dy * dy
}
