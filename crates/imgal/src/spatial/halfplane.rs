use ndarray::{Array1, Array2, ArrayBase, ArrayView1, AsArray, Axis, Ix1, Ix2, ViewRepr, stack};
use rayon::prelude::*;

use crate::prelude::*;
use crate::spatial::convex_hull::graham_scan;

/// Convert the vertices of an edge into halfplane representation.
///
/// # Description
///
/// Converts the two points defining an edge into halfplane representation.
/// The outward-facing line equation is in the form `[Ny, Nx, d]`. The edge
/// vertices are expected to be in `(row, col)` order.
///
/// # Arguments
///
/// * `a`: Vertex `a` of the edge.
/// * `b`: Vertex `b` of the edge.
///
/// # Returns
///
/// * `Ok(Array1<f64>)`: The vector `[Ny, Nx, d]` describing the halfplane.
/// * `Err(ImgalError)`: If points `a` or `b` do not have length `2`.
///
/// # Reference
///
/// * Preparata & Shamos, *Computational geometry an introduction* (1985)\
///   <https://doi.org/10.1007/978-1-4612-1098-6>
#[inline(always)]
pub fn edge_to_halfplane<'a, T, A>(a: A, b: A) -> Result<Array1<f64>, ImgalError>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let a: ArrayBase<ViewRepr<&'a T>, Ix1> = a.into();
    let b: ArrayBase<ViewRepr<&'a T>, Ix1> = b.into();
    if a.len() != 2 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "a",
            expected: 2,
            got: a.len(),
        });
    }
    if b.len() != 2 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "b",
            expected: 2,
            got: b.len(),
        });
    }
    let a_pnt: [f64; 2] = [a[0].to_f64(), a[1].to_f64()];
    let b_pnt: [f64; 2] = [b[0].to_f64(), b[1].to_f64()];
    let py = b_pnt[0] - a_pnt[0];
    let px = b_pnt[1] - a_pnt[1];
    // here we rotate the normal 90 deg clockwise outward
    let ny = -px;
    let nx = py;
    let d = -(ny * a_pnt[0] + nx * a_pnt[1]);
    Ok(Array1::from_vec(vec![ny, nx, d]))
}

/// Compute the intersection of a set of halfplanes.
///
/// # Description
///
/// Computes the convex polygon formed by the intersection of a set of
/// halfplanes. Each halfplane is represented by a row `[Ny, Nx, d]` and
/// contains points satisfying `Ny * y + Nx * x + d < 0`. The interior point
/// *must* lie strictly inside every halfplane. This function shifts the
/// halfplanes relative to the interior point, maps them into "dual space" using
/// line point duality, constructs a convex hull in dual space, and maps the
/// resulting edges back into "primal space" intersection vertices.
///
/// # Arguments
///
/// * `halfplanes`: The halfplanes with `(n_planes, 3)` shape, where each row is
///   `[Ny, Nx, d]`.
/// * `interior_point`: A point with length `2` that lies strictly inside every
///   halfplane and satisfies `Ny * y + Nx * x + d < 0`.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the system's maximum.
///
/// # Returns
///
/// * `Ok(Array2<f64>)`: The vertices of the intersection polygon. The vertices
///   have `(n_points, 2)` shape.
/// * `Err(ImgalError)`: If `halfplanes` is empty. If `halfplanes` axis 1 does
///   not equal `3`. If the interior point length does not equal `2`.
#[inline(always)]
pub fn halfplane_intersection<'a, T, A, B>(
    halfplanes: A,
    interior_point: B,
    threads: Option<usize>,
) -> Result<Array2<f64>, ImgalError>
where
    A: AsArray<'a, f64, Ix2>,
    B: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let halfplanes: ArrayBase<ViewRepr<&'a f64>, Ix2> = halfplanes.into();
    let int_pnt: ArrayBase<ViewRepr<&'a T>, Ix1> = interior_point.into();
    if halfplanes.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "halfplanes",
        });
    }
    if halfplanes.dim().1 != 3 {
        return Err(ImgalError::InvalidAxisLengthExpected {
            arr_name: "halfplanes",
            axis_idx: 1,
            expected: 3,
            got: halfplanes.dim().1,
        });
    }
    let n_hp = halfplanes.dim().0;
    let [qy, qx] = [int_pnt[0].to_f64(), int_pnt[1].to_f64()];
    // we start by converting each halfplane normal vector (primal space) into
    // dual points (dual space)
    let mut dual_points = Array2::<f64>::zeros((n_hp, 2));
    (0..n_hp).try_for_each(|i| {
        let hp = halfplanes.row(i);
        let mut dp = dual_points.row_mut(i);
        let [ny, nx, d] = [hp[0], hp[1], hp[2]];
        let cur_d = ny * qy + nx * qx + d;
        if cur_d.abs() < 1e-12 {
            return Err(ImgalError::InvalidGeneric {
                msg: "The interior point lies on a halfplane boundary.",
            });
        }
        dp[0] = ny / -cur_d;
        dp[1] = nx / -cur_d;
        Ok(())
    })?;
    // constructing a convex hull of the dual points finds the intersection
    // vertices in primal space, the rest of the work is converting dual space
    // back into primal space
    let dual_verts = graham_scan(&dual_points, threads)?;
    let n_dv = dual_verts.dim().0;
    let primal_verts: Vec<f64> = (0..n_dv).fold(Vec::with_capacity(n_dv * 2), |mut acc, i| {
        let a = dual_verts.row(i);
        let b = dual_verts.row((i + 1) % n_dv);
        let [ay, ax] = [a[0], a[1]];
        let [by, bx] = [b[0], b[1]];
        let [dy, dx] = [by - ay, bx - ax];
        let ny = dx;
        let nx = -dy;
        let offset = ny * ay + nx * ax;
        // skip degenerate edages
        if offset.abs() < 1e-12 {
            return acc;
        }
        acc.push((ny / offset) + qy);
        acc.push((nx / offset) + qx);
        acc
    });
    let n_pv = primal_verts.len() / 2;
    if n_pv < 3 {
        return Err(ImgalError::InvalidArrayLengthMinimum {
            arr_name: "primal_verts",
            arr_len: n_pv,
            min_len: 3,
        });
    }
    let primal_verts = Array2::from_shape_vec((n_pv, 2), primal_verts).unwrap();
    graham_scan(primal_verts.view(), threads)
}

/// Convert the vertices of a hull into halfplane representation.
///
/// # Description
///
/// Converts each edge of a hull into halfplane representation. Each edge is
/// converted into an outward-facing line equation in the form `[Ny, Nx, d]`,
/// where each row corresponds to one edge. The vertices are expected to be in
/// `(row, col)` order.
///
/// # Arguments
///
/// * `vertices`: The hull vertices with `(n_points, 2)` shape.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the system's maximum. Parallel computation returns an *unordered* set of
///   halfplanes.
///
/// # Returns
///
/// * `Ok(Array2<f64>)`: The hull in halfplane representation where each row
///   corresponds to one edge.
/// * `Err(ImgalError)`: If `vertices` is empty. If `vertices` axis 1 `!= 2`.
#[inline(always)]
pub fn hull_to_halfplane<'a, T, A>(
    vertices: A,
    threads: Option<usize>,
) -> Result<Array2<f64>, ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    let vertices: ArrayBase<ViewRepr<&'a T>, Ix2> = vertices.into();
    if vertices.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "vertices",
        });
    }
    if vertices.dim().1 != 2 {
        return Err(ImgalError::InvalidAxisLengthExpected {
            arr_name: "vertices",
            axis_idx: 1,
            expected: 2,
            got: vertices.dim().1,
        });
    }
    let n = vertices.dim().0;
    let halfplane_calc = |mut acc: Vec<Array1<f64>>, i: usize| {
        let a = vertices.row(i);
        let b = vertices.row((i + 1) % n);
        // SAFE: this unwrap is safe because we validated the inputs already
        acc.push(edge_to_halfplane(a, b).unwrap());
        acc
    };
    let hp: Vec<Array1<f64>> = par!(threads,
    seq_exp: (0..n).fold(Vec::with_capacity(n), halfplane_calc),
    par_exp: (0..n).into_par_iter().fold(Vec::new, halfplane_calc)
        .reduce(Vec::new, |mut hp_out, hp_thread| {
            hp_out.extend(hp_thread);
            hp_out
        }));
    Ok(stack(
        Axis(0),
        &hp.iter()
            .map(|v| v.view())
            .collect::<Vec<ArrayView1<f64>>>(),
    )
    .unwrap())
}

/// Determine if a query point lies within the intersection of a set of
/// halfplanes.
///
/// # Description
///
/// Determines if the given 2D query point lies within the intersection of *all*
/// the halfplanes. A point is considered inside the halfplane interior if it
/// satisfies `Ny * y + Nx * x + d < 0` for all halfplanes.
///
/// # Arguments
///
/// * `halfplanes`: The halfplanes with `(n_planes, 3)` shape, where each row is
///   `[Ny, Nx, d]`.
/// * `query`: The query point to check if inside a halfplane with
///   `(row, col)` order.
/// * `include_boundary`: If `true` then points on the line boundary are
///   included as valid interior points. If `false` then boundary points are
///   excluded.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the system's maximum.
///
/// # Returns
///
/// * `Ok(bool)`: Returns `true` if `query` is inside all halfplanes, otherwise
///   it returns `false`.
/// * `Err(ImgalError)`: If `halfplanes` is empty. If `halfplanes` axis 1 does
///   not equal `3`. If the query point length does not equal `2`.
#[inline(always)]
pub fn inside_halfplane_interior<'a, T, A, B>(
    halfplanes: A,
    query: B,
    include_boundary: bool,
    threads: Option<usize>,
) -> Result<bool, ImgalError>
where
    A: AsArray<'a, f64, Ix2>,
    B: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let halfplanes: ArrayBase<ViewRepr<&'a f64>, Ix2> = halfplanes.into();
    let query: ArrayBase<ViewRepr<&'a T>, Ix1> = query.into();
    if halfplanes.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "halfplanes",
        });
    }
    if halfplanes.dim().1 != 3 {
        return Err(ImgalError::InvalidAxisLengthExpected {
            arr_name: "halfplanes",
            axis_idx: 1,
            expected: 3,
            got: halfplanes.dim().1,
        });
    }
    if query.len() != 2 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "query",
            expected: 2,
            got: query.len(),
        });
    }
    let [qy, qx] = [query[0].to_f64(), query[1].to_f64()];
    let axis = Axis(0);
    let interior_check = |v: ArrayView1<f64>| v[0] * qy + v[1] * qx + v[2];
    Ok(par!(threads,
    seq_exp: if include_boundary {
        halfplanes.axis_iter(axis).into_iter().all(|v| interior_check(v) <= 0.0)
    } else {
        halfplanes.axis_iter(axis).into_iter().all(|v| interior_check(v) < 0.0)
    },
    par_exp: if include_boundary {
        halfplanes.axis_iter(axis).into_par_iter().all(|v| interior_check(v) <= 0.0)
    } else {
        halfplanes.axis_iter(axis).into_par_iter().all(|v| interior_check(v) < 0.0)
    }))
}
