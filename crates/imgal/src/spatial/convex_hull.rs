use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use ndarray::{Array2, ArrayBase, ArrayView2, AsArray, Axis, Ix2, ViewRepr, s};
use rayon::prelude::*;

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Create a convex hull from a 2D point cloud using Timothy Chan's algorithm.
///
/// # Description
///
/// Constructs a 2D convex hull from a 2D point cloud using Timothy Chan's
/// output-sensitive algorithm. The algorithm iterates with a growing guess *m*
/// for the number of hull vertices *h*. In each phase, the point cloud is
/// partitioned into groups of at most *m* points and a Graham scan is used to
/// create a set of mini-hulls. A Jarvis march is then performed starting from
/// the leftmost point. Each step queries every mini-hull for its right tangent
/// from the current hull vertex and selects the candidate making the smallest
/// clockwise turn as the next hull vertex. If the hull closes within *m* steps
/// the algorithm terminates; otherwise *m* is squared and the algorithm
/// repeats.
///
/// # Arguments
///
/// * `points`: The 2D point cloud with shape `(n_points, 2)`.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Ok(Array2<T>)`: The vertices that comprise the convex hull in
///   clockwise order.
/// * `Err(ImgalError)`: If `points.is_empty() == true`. If the number of points
///   is less than 3.
///
/// # Reference
///
/// <https://en.wikipedia.org/wiki/Chan%27s_algorithm>\
/// <https://doi.org/10.1007%2FBF02712873>
pub fn chan_2d<'a, T, A>(points: A, parallel: bool) -> Result<Array2<T>, ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    let points: ArrayBase<ViewRepr<&'a T>, Ix2> = points.into();
    if points.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "points",
        });
    }
    let n = points.dim().0;
    if n < 3 {
        return Err(ImgalError::InvalidAxisLengthLess {
            arr_name: "points",
            axis_idx: 0,
            value: 3,
        });
    }
    // start by finding the most left (col) point, choosing the lowest point if
    // tied
    let axis = Axis(0);
    let init_idx: usize;
    if parallel {
        init_idx = points
            .axis_iter(axis)
            .enumerate()
            .par_bridge()
            .min_by(|&(_, a), &(_, b)| {
                a[1].partial_cmp(&b[1])
                    .unwrap()
                    .then(a[0].partial_cmp(&b[0]).unwrap())
            })
            .unwrap()
            .0;
    } else {
        init_idx = points
            .axis_iter(axis)
            .enumerate()
            .min_by(|&(_, a), &(_, b)| {
                a[1].partial_cmp(&b[1])
                    .unwrap()
                    .then(a[0].partial_cmp(&b[0]).unwrap())
            })
            .unwrap()
            .0;
    }
    let mut closed: bool = false;
    let mut hull: Vec<[T; 2]> = Vec::new();
    let init_pnt = [points[[init_idx, 0]], points[[init_idx, 1]]];
    for i in 1.. {
        hull.clear();
        let m = get_m(i, n);
        let group_inds = partition_points(n, m);
        let groups: Vec<ArrayView2<T>> = group_inds
            .iter()
            .map(|&(s, e)| points.slice(s![s..e, ..]))
            .collect();
        let group_hulls = groups
            .iter()
            .map(|&g| {
                if g.dim().0 < 3 {
                    Ok(g.to_owned())
                } else {
                    graham_scan(&g, false)
                }
            })
            .collect::<Result<Vec<Array2<T>>, ImgalError>>()?;
        let mut cur_pnt = init_pnt;
        for _ in 0..m {
            hull.push(cur_pnt);
            let mut best_pnt: Option<[T; 2]> = None;
            group_hulls.iter().for_each(|h| {
                if h.is_empty() {
                    return;
                }
                let hn = h.dim().0;
                let cur_pnt_on_hull_idx =
                    (0..hn).find(|&v| h[[v, 0]] == cur_pnt[0] && h[[v, 1]] == cur_pnt[1]);
                let can_pnt = if let Some(v) = cur_pnt_on_hull_idx {
                    let next_idx = (v + 1) % hn;
                    let next_pnt = [h[[next_idx, 0]], h[[next_idx, 1]]];
                    if next_pnt == cur_pnt {
                        return;
                    }
                    next_pnt
                } else {
                    let tan_idx = find_hull_tangent(cur_pnt, h);
                    [h[[tan_idx, 0]], h[[tan_idx, 1]]]
                };
                match best_pnt {
                    Some(b) => {
                        let cross = orient_pred_2d(&cur_pnt, &b, &can_pnt);
                        if cross < -1e-12
                            || (cross.abs() <= 1e-12
                                && dist_sq_2d(&cur_pnt, &can_pnt) > dist_sq_2d(&cur_pnt, &b))
                        {
                            best_pnt = Some(can_pnt);
                        }
                    }
                    None => best_pnt = Some(can_pnt),
                }
            });
            let next_pnt = best_pnt.unwrap_or(init_pnt);
            if next_pnt == init_pnt {
                closed = true;
                break;
            }
            cur_pnt = next_pnt
        }
        if closed {
            break;
        }
    }
    Ok(Array2::from_shape_vec((hull.len(), 2), hull.iter().flat_map(|&p| p).collect()).unwrap())
}

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
/// * `Ok(Array2<T>)`: The vertices that comprise the convex hull in
///   counterclockwise order.
/// * `Err(ImgalError)`: If `points.is_empty() == true`. If the number of points
///   is less than 3.
///
/// # Reference
///
/// <https://en.wikipedia.org/wiki/Graham_scan>\
/// <https://doi.org/10.1016/0020-0190(72)90045-2>
pub fn graham_scan<'a, T, A>(points: A, parallel: bool) -> Result<Array2<T>, ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    let points: ArrayBase<ViewRepr<&'a T>, Ix2> = points.into();
    if points.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "points",
        });
    }
    let n = points.dim().0;
    if n < 3 {
        return Err(ImgalError::InvalidAxisLengthLess {
            arr_name: "points",
            axis_idx: 0,
            value: 3,
        });
    }
    // start by finding the lowest (row) point, choosing the most left if tied
    let axis = Axis(0);
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
    let pivot_pnt = [points[[pivot_idx, 0]], points[[pivot_idx, 1]]];
    let mut point_inds: Vec<usize> = (0..n).map(|i| i).collect();
    point_inds.swap(0, pivot_idx);
    point_inds[1..].sort_by(|&a, &b| {
        let a_pnt = [points[[a, 0]], points[[a, 1]]];
        let b_pnt = [points[[b, 0]], points[[b, 1]]];
        let cross = orient_pred_2d(&pivot_pnt, &a_pnt, &b_pnt);
        if cross.abs() < 1e-12 {
            // points a and b are collinear
            dist_sq_2d(&pivot_pnt, &a_pnt)
                .partial_cmp(&dist_sq_2d(&pivot_pnt, &b_pnt))
                .unwrap()
        } else if cross > 0.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    });
    let hull = point_inds
        .iter()
        .fold(Vec::with_capacity(n), |mut hull: Vec<[T; 2]>, &i| {
            let cur_pnt = [points[[i, 0]], points[[i, 1]]];
            while hull.len() >= 2 {
                let top = hull[hull.len() - 1];
                let second = hull[hull.len() - 2];
                if orient_pred_2d(&second, &top, &cur_pnt) <= 0.0 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(cur_pnt);
            hull
        });
    Ok(Array2::from_shape_vec((hull.len(), 2), hull.iter().flat_map(|&p| p).collect()).unwrap())
}

/// Create a convex hull from a 2D point cloud using the Jarvis march method.
///
/// # Description
///
/// Constructs a 2D convex hull from a 2D point cloud using the Jarvis march
/// method (also known as the "gift wrapping algorithm"). The convex hull is
/// constructed by finding the most left point (col) and iterating through all
/// points in the cloud to find the smallest clockwise trun, from the current
/// position.
///
/// # Arguments
///
/// * `points`: The 2D point cloud with shape `(n_points, 2)`.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Ok(Array2<T>)`: The vertices that comprise the convex hull in clockwise
///   order.
/// * `Err(ImgalError)`: If `points.is_empty() == true`. If the number of points
///   is less than 3.
///
/// # Reference
///
/// <https://en.wikipedia.org/wiki/Gift_wrapping_algorithm>\
/// <https://doi.org/10.1016/0020-0190(73)90020-3>
pub fn jarvis_march<'a, T, A>(points: A, parallel: bool) -> Result<Array2<T>, ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    let points: ArrayBase<ViewRepr<&'a T>, Ix2> = points.into();
    if points.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "points",
        });
    }
    let n = points.dim().0;
    if n < 3 {
        return Err(ImgalError::InvalidAxisLengthLess {
            arr_name: "points",
            axis_idx: 0,
            value: 3,
        });
    }
    // start by finding the most left (col) point, choosing the lowest point if
    // tied
    let axis = Axis(0);
    let init_idx: usize;
    if parallel {
        init_idx = points
            .axis_iter(axis)
            .enumerate()
            .par_bridge()
            .min_by(|&(_, a), &(_, b)| {
                a[1].partial_cmp(&b[1])
                    .unwrap()
                    .then(a[0].partial_cmp(&b[0]).unwrap())
            })
            .unwrap()
            .0;
    } else {
        init_idx = points
            .axis_iter(axis)
            .enumerate()
            .min_by(|&(_, a), &(_, b)| {
                a[1].partial_cmp(&b[1])
                    .unwrap()
                    .then(a[0].partial_cmp(&b[0]).unwrap())
            })
            .unwrap()
            .0;
    }
    let mut hull: Vec<[T; 2]> = Vec::new();
    let mut cur_idx = init_idx;
    loop {
        let cur_pnt = [points[[cur_idx, 0]], points[[cur_idx, 1]]];
        hull.push(cur_pnt);
        let mut best_idx = (cur_idx + 1) % n;
        (0..n).for_each(|i| {
            if i == cur_idx {
                return;
            }
            let next_pnt = [points[[best_idx, 0]], points[[best_idx, 1]]];
            let i_pnt = [points[[i, 0]], points[[i, 1]]];
            let cross = orient_pred_2d(&cur_pnt, &next_pnt, &i_pnt);
            if cross < -1e-12
                || (cross.abs() <= 1e-12)
                    && dist_sq_2d(&cur_pnt, &i_pnt) > dist_sq_2d(&cur_pnt, &next_pnt)
            {
                best_idx = i;
            }
        });
        cur_idx = best_idx;
        if cur_idx == init_idx || hull.len() > n {
            break;
        }
    }
    Ok(Array2::from_shape_vec((hull.len(), 2), hull.iter().flat_map(|&p| p).collect()).unwrap())
}

/// TODO
pub fn quick_hull_3d<'a, T, A>(
    points: A,
    parallel: bool,
) -> Result<(Array2<T>, Vec<[usize; 3]>), ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    let points: ArrayBase<ViewRepr<&'a T>, Ix2> = points.into();
    if points.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "points",
        });
    }
    let n = points.dim().0;
    if n < 4 {
        return Err(ImgalError::InvalidAxisLengthLess {
            arr_name: "points",
            axis_idx: 0,
            value: 4,
        });
    }
    let pnts: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            [
                points[[i, 0]].to_f64(),
                points[[i, 1]].to_f64(),
                points[[i, 2]].to_f64(),
            ]
        })
        .collect();
    let pa = (0..n)
        .min_by(|&a, &b| pnts[a][2].partial_cmp(&pnts[b][2]).unwrap())
        .unwrap();
    let pb = (0..n)
        .max_by(|&a, &b| pnts[a][2].partial_cmp(&pnts[b][2]).unwrap())
        .unwrap();
    let pc = (0..n)
        .filter(|&i| i != pa && i != pb)
        .max_by(|&a, &b| {
            triangle_area_sq(&pnts[pa], &pnts[pb], &pnts[a])
                .partial_cmp(&triangle_area_sq(&pnts[pa], &pnts[pb], &pnts[b]))
                .unwrap()
        })
        .ok_or(ImgalError::InvalidAxisLengthLess {
            arr_name: "points",
            axis_idx: 0,
            value: 4,
        })?;
    let pd = (0..n)
        .filter(|&i| i != pa && i != pb && i != pc)
        .max_by(|&a, &b| {
            orient_pred_3d(&pnts[pa], &pnts[pb], &pnts[pc], &pnts[a])
                .abs()
                .partial_cmp(&orient_pred_3d(&pnts[pa], &pnts[pb], &pnts[pc], &pnts[b]).abs())
                .unwrap()
        })
        .ok_or(ImgalError::InvalidAxisLengthLess {
            arr_name: "points",
            axis_idx: 0,
            value: 4,
        })?;
    let interior = [
        (pnts[pa][0] + pnts[pb][0] + pnts[pc][0] + pnts[pd][0]) / 4.0,
        (pnts[pa][1] + pnts[pb][1] + pnts[pc][1] + pnts[pd][1]) / 4.0,
        (pnts[pa][2] + pnts[pb][2] + pnts[pc][2] + pnts[pd][2]) / 4.0,
    ];
    let mut faces: Vec<[usize; 3]> = vec![
        flip_face_out(&pnts, [pa, pb, pc], &interior),
        flip_face_out(&pnts, [pa, pb, pd], &interior),
        flip_face_out(&pnts, [pb, pc, pd], &interior),
        flip_face_out(&pnts, [pa, pc, pd], &interior),
    ];
    let mut outside: Vec<Vec<usize>> = faces
        .iter()
        .map(|f| {
            (0..n)
                .filter(|&i| {
                    i != f[0]
                        && i != f[1]
                        && i != f[2]
                        && orient_pred_3d(&pnts[f[0]], &pnts[f[1]], &pnts[f[2]], &pnts[i]) > 1e-12
                })
                .collect()
        })
        .collect();
    loop {
        let Some(fi) = outside.iter().position(|o| !o.is_empty()) else {
            break;
        };
        let apex = *outside[fi]
            .iter()
            .max_by(|&&a, &&b| {
                orient_pred_3d(
                    &pnts[faces[fi][0]],
                    &pnts[faces[fi][1]],
                    &pnts[faces[fi][2]],
                    &pnts[a],
                )
                .partial_cmp(&orient_pred_3d(
                    &pnts[faces[fi][0]],
                    &pnts[faces[fi][1]],
                    &pnts[faces[fi][2]],
                    &pnts[b],
                ))
                .unwrap()
            })
            .unwrap();
        let visible: HashSet<usize>;
        if parallel {
            visible = (0..faces.len())
                .into_par_iter()
                .filter(|&i| {
                    orient_pred_3d(
                        &pnts[faces[i][0]],
                        &pnts[faces[i][1]],
                        &pnts[faces[i][2]],
                        &pnts[apex],
                    ) > 1e-12
                })
                .collect();
        } else {
            visible = (0..faces.len())
                .filter(|&i| {
                    orient_pred_3d(
                        &pnts[faces[i][0]],
                        &pnts[faces[i][1]],
                        &pnts[faces[i][2]],
                        &pnts[apex],
                    ) > 1e-12
                })
                .collect();
        }
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
        visible.iter().for_each(|&i| {
            let f = faces[i];
            for edge in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        });
        let horizon: Vec<(usize, usize)> = edge_count
            .keys()
            .filter(|&&(u, v)| !edge_count.contains_key(&(v, u)))
            .copied()
            .collect();
        let orphans: Vec<usize> = {
            let mut seen = HashSet::new();
            visible
                .iter()
                .flat_map(|&vi| outside[vi].iter().copied())
                .filter(|&i| i != apex && seen.insert(i))
                .collect()
        };
        let new_faces: Vec<[usize; 3]> = horizon
            .iter()
            .map(|&(u, v)| flip_face_out(&pnts, [apex, u, v], &interior))
            .collect();
        let mut to_remove: Vec<usize> = visible.into_iter().collect();
        to_remove.sort_unstable_by(|a, b| b.cmp(a));
        to_remove.iter().for_each(|&i| {
            faces.swap_remove(i);
            outside.swap_remove(i);
        });
        new_faces.iter().for_each(|&f| {
            let o: Vec<usize> = orphans
                .iter()
                .copied()
                .filter(|&i| orient_pred_3d(&pnts[f[0]], &pnts[f[1]], &pnts[f[2]], &pnts[i]) > 1e-12)
                .collect();
            faces.push(f);
            outside.push(o);
        });
    }

    let seen: Vec<usize> = {
        let mut set = HashSet::new();
        let mut v: Vec<usize> = faces
            .iter()
            .flat_map(|f| f.iter().copied())
            .filter(|&i| set.insert(i))
            .collect();
        v.sort_unstable();
        v
    };
    let mut remap = vec![0_usize; n];
    seen.iter()
        .enumerate()
        .for_each(|(new, &old)| remap[old] = new);
    let mut hull_vertices = Array2::<T>::default((seen.len(), 3));
    seen.iter().enumerate().for_each(|(new, &old)| {
        hull_vertices[[new, 0]] = points[[old, 0]];
        hull_vertices[[new, 1]] = points[[old, 1]];
        hull_vertices[[new, 2]] = points[[old, 2]];
    });
    let faces: Vec<[usize; 3]> = faces
        .into_iter()
        .map(|f| [remap[f[0]], remap[f[1]], remap[f[2]]])
        .collect();
    Ok((hull_vertices, faces))
}

/// Compute the squared Euclidean distance between two 2D points.
///
/// # Arguments
///
/// * `point_a`: The first point as (row, col).
/// * `point_b`: The second point as (row, col).
///
/// # Returns
///
/// * `T`: The squared Euclidean distance.
pub fn dist_sq_2d<T>(point_a: &[T; 2], b: &[T; 2]) -> T
where
    T: AsNumeric,
{
    let dy = point_a[0] - b[0];
    let dx = point_a[1] - b[1];
    dx * dx + dy * dy
}

/// Find the right tangent point on a counterclockwise convex hull from an
/// external query point using binary search.
///
/// # Arguments
///
/// * `query_point`: The coordinates for the query point.
/// * `hull`: The convex hull to search in counterclockwise order.
///
/// # Returns
///
/// * `usize`: The right tangent point index on the convex hull relative to
///   the query point.
fn find_hull_tangent<'a, T, A>(query_point: [T; 2], hull: A) -> usize
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    let hull: ArrayBase<ViewRepr<&'a T>, Ix2> = hull.into();
    let n = hull.dim().0;
    if n == 1 {
        return 0;
    }
    if n == 2 {
        let point_a = [hull[[0, 0]], hull[[0, 1]]];
        let point_b = [hull[[1, 0]], hull[[1, 1]]];
        let cross = orient_pred_2d(&query_point, &point_a, &point_b);
        return if cross < -1e-12
            || (cross.abs() <= 1e-12
                && dist_sq_2d(&query_point, &point_b) > dist_sq_2d(&query_point, &point_a))
        {
            1
        } else {
            0
        };
    }
    let edge_cross = |i: usize| -> f64 {
        let a_idx = i % n;
        let b_idx = (i + 1) % n;
        let point_a = [hull[[a_idx, 0]], hull[[a_idx, 1]]];
        let point_b = [hull[[b_idx, 0]], hull[[b_idx, 1]]];
        orient_pred_2d(&query_point, &point_a, &point_b)
    };
    let point_to_point_cross = |i: usize, j: usize| -> f64 {
        let i_idx = i % n;
        let j_idx = j % n;
        let point_a = [hull[[i_idx, 0]], hull[[i_idx, 1]]];
        let point_b = [hull[[j_idx, 0]], hull[[j_idx, 1]]];
        orient_pred_2d(&query_point, &point_a, &point_b)
    };
    let mut lo: usize = 0;
    let mut hi = n;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        let is_lo_up = edge_cross(lo) >= 0.0;
        let is_mid_up = edge_cross(mid) >= 0.0;
        let compare = point_to_point_cross(lo, mid);
        if is_lo_up {
            if is_mid_up {
                if compare < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            } else {
                lo = mid;
            }
        } else {
            if is_mid_up {
                hi = mid;
            } else {
                if compare < 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
        }
    }
    if edge_cross(lo) >= 0.0 { lo } else { hi % n }
}

/// TODO
#[inline]
fn flip_face_out(points: &[[f64; 3]], face: [usize; 3], inside_point: &[f64; 3]) -> [usize; 3] {
    if orient_pred_3d(
        &points[face[0]],
        &points[face[1]],
        &points[face[2]],
        inside_point,
    ) > 0.0
    {
        [face[0], face[2], face[1]]
    } else {
        face
    }
}

/// Compute the `m` value at iteration `i`.
///
/// # Arguments
///
/// * `i`: The current loop iteration.
/// * `n`: The number of points in the point cloud.
///
/// # Returns
///
/// * `usize`: The `m` value for Chan's algorithm (*i.e.* the guessed hull
/// size) cappepd at size `n`.
fn get_m(i: i32, n: usize) -> usize {
    if i >= 20 {
        return n;
    }
    let exponent: u64 = 1 << i;
    if exponent >= 64 {
        return n;
    }
    let m: usize = 1 << exponent;
    m.min(n)
}

/// Compute the 2D cross product of vectors defined by three points. This
/// function is also known as the orientiation predicate for the 2D case.
///
/// # Description
///
/// Calculates the cross product of vectors `(a - o)` and `(b - o)`. The result
/// indicates rotational direction:
///
/// - Positive => counterclockwise (left) turn
/// - Negative => clockwise (right) turn
/// - Zero => collinear points
///
/// # Arguments
///
/// * `o`: The origin point as (row, col).
/// * `a`: The first point as (row, col).
/// * `b`: The second point as (row, col).
///
/// # Returns
///
/// * `T`: The cross product.
///
/// # Reference
///
/// <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>
pub fn orient_pred_2d<T>(o: &[T; 2], a: &[T; 2], b: &[T; 2]) -> f64
where
    T: AsNumeric,
{
    (a[1].to_f64() - o[1].to_f64()) * (b[0].to_f64() - o[0].to_f64())
        - (a[0].to_f64() - o[0].to_f64()) * (b[1].to_f64() - o[1].to_f64())
}

/// Computes the 3D signed volume (*i.e.* orientation) of a tetrahedron.
///
/// # Description
///
/// The sign indicates the tetrahedron orientation:
/// - Positive => Point `d` is below the plane in CCW from outside the hull.
/// - Negative => Point `d` is above the plane in CCW from outside the hull.
/// - Zero => Point `d` lines on the plane (coplanar).
///
/// Note that this function assumes a "right-handed" system (X, Y, Z) which
/// means need to take the opposite sign when working in the "left-handed"
/// system (pln, row, col).
///
/// # Returns
///
/// * `f64`: The orientation of the tetrahedron.
///
/// # Reference
///
/// <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>
/// <https://doi.org/10.1007/PL00009321>
#[inline]
fn orient_pred_3d(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3], d: &[f64; 3]) -> f64 {
    let [adx, ady, adz] = [a[2] - d[2], a[1] - d[1], a[0] - d[0]];
    let [bdx, bdy, bdz] = [b[2] - d[2], b[1] - d[1], b[0] - d[0]];
    let [cdx, cdy, cdz] = [c[2] - d[2], c[1] - d[1], c[0] - d[0]];
    adx * (bdy * cdz - bdz * cdy) - ady * (bdx * cdz - bdz * cdx) + adz * (bdx * cdy - bdy * cdx)
}

/// Create mini-hull partition start and end intervals.
///
/// # Returns
///
/// * `Vec<(usize, usize)>`: The start and end values for paritions.
fn partition_points(n_points: usize, m: usize) -> Vec<(usize, usize)> {
    let mut partitions = Vec::new();
    let mut start = 0;
    while start < n_points {
        let end = (start + m).min(n_points);
        partitions.push((start, end));
        start = end;
    }
    partitions
}

/// Computes the squared area of the triangle defined by three 3D points `a`,
/// `b`, and `c` by taking the cross product of the edge vectors `ab = b - a`
/// `ac = c - a`.
///
/// # Returns
///
/// * `f64`: The squared area of the triangle (*i.e.* `4 * (area)^2`).
#[inline]
fn triangle_area_sq(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    let [abx, aby, abz] = [b[2] - a[2], b[1] - a[1], b[0] - a[0]];
    let [acx, acy, acz] = [c[2] - a[2], c[1] - a[1], c[0] - a[0]];
    ((aby * acz - abz * acy) * (aby * acz - abz * acy))
        + ((abz * acx - abx * acz) * (abz * acx - abx * acz)
            + ((abx * acy - aby * acx) * (abx * acy - aby * acx)))
}
