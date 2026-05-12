use std::array;

use ndarray::{Array1, Array2, ArrayBase, ArrayView1, AsArray, Axis, Ix1, Ix2, ViewRepr, stack};

use crate::error::ImgalError;
use crate::spatial::convex_hull::quickhull_3d;
use crate::traits::numeric::AsNumeric;

/// Compute the intersection of a set of halfspaces.
///
/// # Description
///
/// todo...I think I might make this fn accept vertices of a convex hull.
///
/// # Arguments
///
/// * `halfspaces`: A set of halfspaces in shape `(n_spaces, 4)`.
/// * `interior_point`: The point lies inside every halfspace and satisfies
///   `Nz * z + Ny * y + Nx * x + d < 0`.
///
/// # Returns
///
/// * `Ok()`:
/// * `Err()`:
pub fn halfspace_intersection<'a, T, A, B>(
    halfspaces: A,
    interior_point: B,
) -> Result<(Array2<T>, Array2<usize>), ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    B: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let halfspaces: ArrayBase<ViewRepr<&'a T>, Ix2> = halfspaces.into();
    let in_pnt: ArrayBase<ViewRepr<&'a T>, Ix1> = interior_point.into();
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
    if in_pnt.len() != 3 {
        return Err(ImgalError::InvalidAxisLengthExpected {
            arr_name: "interior_point",
            axis_idx: 1,
            expected: 3,
            got: in_pnt.len(),
        });
    }
    let n_h = halfspaces.dim().0;
    let [iz, iy, ix] = array::from_fn(|i| in_pnt[i].to_f64());
    // we start by convert each halfspace normal vector (primal space) into dual
    // points (dual space)
    let mut dual_points = Array2::<f64>::zeros((n_h, 3));
    (0..n_h).try_for_each(|i| {
        let [nz, ny, nx, d] = array::from_fn(|j| halfspaces[[i, j]].to_f64());
        let d_prime = nz * iz + ny * iy + nx * ix + d;
        if d_prime.abs() < 1e-12 {
            return Err(ImgalError::InvalidGeneric {
                msg: "The interior point lies on a halfspace boundary.",
            });
        }
        dual_points[[i, 0]] = nz / -d_prime;
        dual_points[[i, 1]] = ny / -d_prime;
        dual_points[[i, 2]] = nx / -d_prime;
        Ok(())
    })?;
    // constructing convex hull of dual points finds the intersection vertices
    // in primal space after converting back
    let (dual_verts, dual_faces) = quickhull_3d(&dual_points, false)?;
    let n_df = dual_faces.dim().0;
    let primal_verts: Vec<T> = (0..n_df).fold(Vec::with_capacity(n_df * 3), |mut acc, i| {
        let [a_idx, b_idx, c_idx] = array::from_fn(|j| dual_faces[[i, j]]);
        let [az, ay, ax] = array::from_fn(|j| dual_verts[[a_idx, j]]);
        let [bz, by, bx] = array::from_fn(|j| dual_verts[[b_idx, j]]);
        let [cz, cy, cx] = array::from_fn(|j| dual_verts[[c_idx, j]]);
        let [baz, bay, bax] = [bz - az, by - ay, bx - ax];
        let [caz, cay, cax] = [cz - az, cy - ay, cx - ax];
        let nz = bax * cay - bay * cax;
        let ny = baz * cax - bax * caz;
        let nx = bay * caz - baz * cay;
        let offset = nz * az + ny * ay + nx * ax;
        // skip degenerate planes
        if offset.abs() < 1e-12 {
            return acc;
        }
        acc.push(T::from_f64((nz / offset) + iz));
        acc.push(T::from_f64((ny / offset) + iy));
        acc.push(T::from_f64((nx / offset) + ix));
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
    quickhull_3d(&primal_verts, false)
}

/// Convert the vertices of a tetrahedron face into halfspace representation.
///
/// # Description
///
/// Converts the three points defining a face of a tetrahedron (*i.e.* a
/// triangle) into half halfspace representation. The outward-facing plane
/// equation is in the form `[Nz, Ny, Nx, d]`. The triangle vertices are
/// expected to be in `(pln, row, col)` order.
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
/// * `Err(ImgalError)`: If point arrays `a`, `b`, or `c` do not equal `3`.
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
    let a_pnt: [f64; 3] = array::from_fn(|i| a[i].to_f64());
    let b_pnt: [f64; 3] = array::from_fn(|i| b[i].to_f64());
    let c_pnt: [f64; 3] = array::from_fn(|i| c[i].to_f64());
    let [pz, py, px] = array::from_fn(|i| b_pnt[i] - a_pnt[i]);
    let [qz, qy, qx] = array::from_fn(|i| c_pnt[i] - a_pnt[i]);
    let nz = -(py * qx - px * qy);
    let ny = -(px * qz - pz * qx);
    let nx = -(pz * qy - py * qz);
    let d = -(a_pnt[0] * nz + a_pnt[1] * ny + a_pnt[2] * nx);
    Ok(Array1::from_iter([nz, ny, nx, d]))
}

/// Convert the vertices and triangular faces of a hull into halfspace
/// representation.
///
/// # Description
///
/// Converts each triangular face of a convex hull into halfspace
/// representation. Each face is converted into an outward-facing plane equation
/// in the form `[Nz, Ny, Nx, d]`, where each row corresponds to one face. The
/// vertices are expected to be in `(pln, row, col)` order.
///
/// # Arguments
///
/// * `vertices`: The hull vertices with `(n_points, 3)` shape.
/// * `faces`: The hull faces with `(n_triangle, 3)` shape.
///
/// # Returns
///
/// * `Ok(Array2<f64>)`: The hull in halfspace representation where each row
///   corresponds to one face.
/// * `Err(ImgalError)`: If `vertices` and/or `faces` is empty. If `vertices`
///   and/or `faces` axis 1 `!= 3`.
#[inline]
pub fn hull_to_halfspace<'a, T, A, B>(vertices: A, faces: B) -> Result<Array2<f64>, ImgalError>
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
    let hs: Vec<Array1<f64>> = (0..n).try_fold(Vec::with_capacity(n), |mut acc, i| {
        let [a_idx, b_idx, c_idx] = array::from_fn(|j| faces[[i, j]]);
        acc.push(face_to_halfspace(
            vertices.row(a_idx),
            vertices.row(b_idx),
            vertices.row(c_idx),
        )?);
        Ok(acc)
    })?;
    Ok(stack(
        Axis(0),
        &hs.iter()
            .map(|v| v.view())
            .collect::<Vec<ArrayView1<f64>>>(),
    )
    .unwrap())
}
