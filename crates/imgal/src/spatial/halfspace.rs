use std::array;

use ndarray::{Array1, Array2, ArrayBase, ArrayView1, AsArray, Axis, Ix1, Ix2, ViewRepr, stack};
use rayon::prelude::*;

use crate::prelude::*;
use crate::spatial::convex_hull::quickhull_3d;

/// Convert the vertices of a tetrahedron face into halfspace representation.
///
/// # Description
///
/// Converts the three points defining a face of a tetrahedron (*i.e.* a
/// triangle) into halfspace representation. The outward-facing plane equation
/// is in the form `[Nz, Ny, Nx, d]`. The triangle vertices are expected to be
/// in `(pln, row, col)` order.
///
/// # Arguments
///
/// * `a`: Vertex `a` of the triangle face.
/// * `b`: Vertex `b` of the triangle face.
/// * `c`: Vertex `c` of the triangle face.
///
/// # Returns
///
/// * `Ok(Array1<f64>)`: The vector `[Nz, Ny, Nx, d]` describing the halfspace.
/// * `Err(ImgalError)`: If points `a`, `b`, or `c` do not have length `3`.
#[inline]
pub fn face_to_halfspace<'a, T, A>(a: A, b: A, c: A) -> Result<Array1<f64>, ImgalError>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let a: ArrayBase<ViewRepr<&'a T>, Ix1> = a.into();
    let b: ArrayBase<ViewRepr<&'a T>, Ix1> = b.into();
    let c: ArrayBase<ViewRepr<&'a T>, Ix1> = c.into();
    if a.len() != 3 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "a",
            expected: 3,
            got: a.len(),
        });
    }
    if b.len() != 3 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "b",
            expected: 3,
            got: b.len(),
        });
    }
    if c.len() != 3 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "c",
            expected: 3,
            got: c.len(),
        });
    }
    let a_pnt: [f64; 3] = [a[0].to_f64(), a[1].to_f64(), a[2].to_f64()];
    let b_pnt: [f64; 3] = [b[0].to_f64(), b[1].to_f64(), b[2].to_f64()];
    let c_pnt: [f64; 3] = [c[0].to_f64(), c[1].to_f64(), c[2].to_f64()];
    let [pz, py, px] = array::from_fn(|i| b_pnt[i] - a_pnt[i]);
    let [qz, qy, qx] = array::from_fn(|i| c_pnt[i] - a_pnt[i]);
    let nz = py * qx - px * qy;
    let ny = px * qz - pz * qx;
    let nx = pz * qy - py * qz;
    let d = -(a_pnt[0] * nz + a_pnt[1] * ny + a_pnt[2] * nx);
    Ok(Array1::from_vec(vec![nz, ny, nx, d]))
}

/// Compute the intersection of a set of halfspaces.
///
/// # Description
///
/// Computes the convex polyhedron formed by the intersection of a set of
/// halfspaces. Each halfspace is represented by a row `[Nz, Ny, Nx, d]` and
/// contains points satisfying `Nz * z + Ny * y + Nx * x + d < 0`. The interior
/// point *must* lie strictly inside every halfspace. This function shifts the
/// halfspaces relative to the interior point, maps them into "dual space" using
/// line point duality, constructs a convex hull in dual space, and maps the
/// resulting faces back into "primal space" intersection vertices.
///
/// # Arguments
///
/// * `halfspaces`: The halfspaces with `(n_spaces, 4)` shape, where each row is
///   `[Nz, Ny, Nx, d]`.
/// * `interior_point`: A point with length `3` that lies strictly inside every
///   halfspace and satisfies `Nz * z + Ny * y + Nx * x + d < 0`.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `Ok((Array2<f64>, Array2<usize>))`: The vertices and triangular faces of
///   the intersection polyhedron. The vertices have `(n_points, 3)` shape and
///   the faces have `(n_triangles, 3)` shape.
/// * `Err(ImgalError)`: If `halfspaces` is empty. If `halfspaces` axis 1 does
///   not equal `4`. If the interior point length does not equal `3`.
#[inline]
pub fn halfspace_intersection<'a, T, A, B>(
    halfspaces: A,
    interior_point: B,
    threads: Option<usize>,
) -> Result<(Array2<f64>, Array2<usize>), ImgalError>
where
    A: AsArray<'a, f64, Ix2>,
    B: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let halfspaces: ArrayBase<ViewRepr<&'a f64>, Ix2> = halfspaces.into();
    let int_pnt: ArrayBase<ViewRepr<&'a T>, Ix1> = interior_point.into();
    if halfspaces.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "halfspaces",
        });
    }
    if halfspaces.dim().1 != 4 {
        return Err(ImgalError::InvalidAxisLengthExpected {
            arr_name: "halfspaces",
            axis_idx: 1,
            expected: 4,
            got: halfspaces.dim().1,
        });
    }
    if int_pnt.len() != 3 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "interior_point",
            expected: 3,
            got: int_pnt.len(),
        });
    }
    let n_h = halfspaces.dim().0;
    let [qz, qy, qx] = [
        int_pnt[0].to_f64(),
        int_pnt[1].to_f64(),
        int_pnt[2].to_f64(),
    ];
    // we start by convert each halfspace normal vector (primal space) into dual
    // points (dual space)
    let mut dual_points = Array2::<f64>::zeros((n_h, 3));
    (0..n_h).try_for_each(|i| {
        let [nz, ny, nx, d] = [
            halfspaces[[i, 0]],
            halfspaces[[i, 1]],
            halfspaces[[i, 2]],
            halfspaces[[i, 3]],
        ];
        let cur_d = nz * qz + ny * qy + nx * qx + d;
        if cur_d.abs() < 1e-12 {
            return Err(ImgalError::InvalidGeneric {
                msg: "The interior point lies on a halfspace boundary.",
            });
        }
        dual_points[[i, 0]] = nz / -cur_d;
        dual_points[[i, 1]] = ny / -cur_d;
        dual_points[[i, 2]] = nx / -cur_d;
        Ok(())
    })?;
    // constructing convex hull of dual points finds the intersection vertices
    // in primal space after converting back
    let (dual_verts, dual_faces) = quickhull_3d(&dual_points, threads)?;
    let n_df = dual_faces.dim().0;
    let primal_verts: Vec<f64> = (0..n_df).fold(Vec::with_capacity(n_df * 3), |mut acc, i| {
        let [a_idx, b_idx, c_idx] = [dual_faces[[i, 0]], dual_faces[[i, 1]], dual_faces[[i, 2]]];
        let [az, ay, ax] = [
            dual_verts[[a_idx, 0]],
            dual_verts[[a_idx, 1]],
            dual_verts[[a_idx, 2]],
        ];
        let [bz, by, bx] = [
            dual_verts[[b_idx, 0]],
            dual_verts[[b_idx, 1]],
            dual_verts[[b_idx, 2]],
        ];
        let [cz, cy, cx] = [
            dual_verts[[c_idx, 0]],
            dual_verts[[c_idx, 1]],
            dual_verts[[c_idx, 2]],
        ];
        let [zba, yba, xba] = [bz - az, by - ay, bx - ax];
        let [zca, yca, xca] = [cz - az, cy - ay, cx - ax];
        let nz = xba * yca - yba * xca;
        let ny = zba * xca - xba * zca;
        let nx = yba * zca - zba * yca;
        let offset = nz * az + ny * ay + nx * ax;
        // skip degenerate planes
        if offset.abs() < 1e-12 {
            return acc;
        }
        acc.push((nz / offset) + qz);
        acc.push((ny / offset) + qy);
        acc.push((nx / offset) + qx);
        acc
    });
    let n_pv = primal_verts.len() / 3;
    if n_pv < 4 {
        return Err(ImgalError::InvalidArrayLengthMinimum {
            arr_name: "primal_verts",
            arr_len: n_pv,
            min_len: 4,
        });
    }
    let primal_verts = Array2::from_shape_vec((n_pv, 3), primal_verts).unwrap();
    quickhull_3d(&primal_verts, threads)
}

/// Convert the vertices and triangular faces of a hull into halfspace
/// representation.
///
/// # Description
///
/// Converts each triangular face of a hull into halfspace representation. Each
/// face is converted into an outward-facing plane equation in the form
/// `[Nz, Ny, Nx, d]`, where each row corresponds to one face. The vertices are
/// expected to be in `(pln, row, col)` order.
///
/// # Arguments
///
/// * `vertices`: The hull vertices with `(n_points, 3)` shape.
/// * `faces`: The hull faces with `(n_triangle, 3)` shape.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum. Parallel computation returns an *unordered* set of
///   halfspaces.
///
/// # Returns
///
/// * `Ok(Array2<f64>)`: The hull in halfspace representation where each row
///   corresponds to one face.
/// * `Err(ImgalError)`: If `vertices` and/or `faces` is empty. If `vertices`
///   and/or `faces` axis 1 `!= 3`.
#[inline]
pub fn hull_to_halfspace<'a, T, A, B>(
    vertices: A,
    faces: B,
    threads: Option<usize>,
) -> Result<Array2<f64>, ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    B: AsArray<'a, usize, Ix2>,
    T: 'a + AsNumeric,
{
    let vertices: ArrayBase<ViewRepr<&'a T>, Ix2> = vertices.into();
    let faces: ArrayBase<ViewRepr<&'a usize>, Ix2> = faces.into();
    if vertices.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "vertices",
        });
    }
    if vertices.dim().1 != 3 {
        return Err(ImgalError::InvalidAxisLengthExpected {
            arr_name: "vertices",
            axis_idx: 1,
            expected: 3,
            got: vertices.dim().1,
        });
    }
    if faces.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "faces",
        });
    }
    if faces.dim().1 != 3 {
        return Err(ImgalError::InvalidAxisLengthExpected {
            arr_name: "faces",
            axis_idx: 1,
            expected: 3,
            got: faces.dim().1,
        });
    }
    let n = faces.dim().0;
    let halfspace_calc = |mut acc: Vec<Array1<f64>>, i: usize| {
        let [a_idx, b_idx, c_idx] = [faces[[i, 0]], faces[[i, 1]], faces[[i, 2]]];
        // SAFE: this unwrap is safe because we validated the inputs already
        acc.push(
            face_to_halfspace(
                vertices.row(a_idx),
                vertices.row(b_idx),
                vertices.row(c_idx),
            )
            .unwrap(),
        );
        acc
    };
    let hs: Vec<Array1<f64>> = par!(threads,
    seq_exp: (0..n).fold(Vec::with_capacity(n), halfspace_calc),
    par_exp: (0..n).into_par_iter().fold(Vec::new, halfspace_calc)
        .reduce(Vec::new, |mut hs_out, hs_thread| {
            hs_out.extend(hs_thread);
            hs_out
        }));
    Ok(stack(
        Axis(0),
        &hs.iter()
            .map(|v| v.view())
            .collect::<Vec<ArrayView1<f64>>>(),
    )
    .unwrap())
}

/// Determine if a query point lies within the intersection of a set of
/// halfspaces.
///
/// # Description
///
/// Determines if the given 3D query point lies within the intersection of *all*
/// the halfspaces. A point is considered inside the halfspace interior if it
/// satisfies `Nz * z + Ny * y + Nx * x + d < 0` for all halfspaces.
///
/// # Arguments
///
/// * `halfspaces`: The halfspaces with `(n_spaces, 4)` shape, where each row is
///   `[Nz, Ny, Nx, d]`.
/// * `query`: The query point to check if inside a halfspace with
///   `(pln, row, col)` order.
/// * `include_boundary`: If `true` then points on the face boundary are
///   included as valid interior points. If `false` then boundary points are
///   excluded.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `Ok(bool)`: Returns `true` if `query` is inside all halfspaces, otherwise
///   it returns `false`.
/// * `Err(ImgalError)`: If `halfspaces` is empty. If `halfspaces` axis 1 does
///   not equal `4`. If the query point length does not equal `3`.
#[inline]
pub fn inside_halfspace_interior<'a, T, A, B>(
    halfspaces: A,
    query: B,
    include_boundary: bool,
    threads: Option<usize>,
) -> Result<bool, ImgalError>
where
    A: AsArray<'a, f64, Ix2>,
    B: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let halfspaces: ArrayBase<ViewRepr<&'a f64>, Ix2> = halfspaces.into();
    let query: ArrayBase<ViewRepr<&'a T>, Ix1> = query.into();
    if halfspaces.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "halfspaces",
        });
    }
    if halfspaces.dim().1 != 4 {
        return Err(ImgalError::InvalidAxisLengthExpected {
            arr_name: "halfspaces",
            axis_idx: 1,
            expected: 4,
            got: halfspaces.dim().1,
        });
    }
    if query.len() != 3 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "query",
            expected: 3,
            got: query.len(),
        });
    }
    let [qz, qy, qx] = [query[0].to_f64(), query[1].to_f64(), query[2].to_f64()];
    let axis = Axis(0);
    let interior_check = |v: ArrayView1<f64>| v[0] * qz + v[1] * qy + v[2] * qx + v[3];
    Ok(par!(threads,
    seq_exp: if include_boundary {
        halfspaces.axis_iter(axis).into_iter().all(|v| interior_check(v) <= 0.0)
    } else {
        halfspaces.axis_iter(axis).into_iter().all(|v| interior_check(v) < 0.0)
    },
    par_exp: if include_boundary {
        halfspaces.axis_iter(axis).into_par_iter().all(|v| interior_check(v) <= 0.0)
    } else {
        halfspaces.axis_iter(axis).into_par_iter().all(|v| interior_check(v) < 0.0)
    }))
}
