use ndarray::{Array1, ArrayBase, AsArray, Ix1, Ix2, ViewRepr};
use rayon::prelude::*;

use crate::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Determine if a query point is inside a polyhedron.
///
/// # Description
///
/// Determines if a 3D query point is inside the given polyhedron's interior.
/// Each face of the polyhedron is used to form a tetrahedron with the `center`
/// point. The query point is considered inside the polyhedron if it is inside
/// one of the constituent tetrahedra. The function expects points in
/// `(pln, row, col)` order.
///
/// # Arguments
///
/// * `vertices`: The hull vertices with `(n_points, 3)` shape.
/// * `faces`: The hull faces with `(n_triangle, 3)` shape.
/// * `center`: The center point of the polyhedron.
/// * `query`: The query point to check if inside the polyhedron.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Ok(bool)`: Returns `true` if `query` is inside the polyhedron, otherwise
///   it returns `false`.
/// * `Err(ImgalError)`: If `vertices` and/or `faces` is empty. If `vertices`
///   and/or `faces` axis 1 `!= 3`. If `center` or `query` length does not equal
///   `3`.
pub fn inside_polyhedron<'a, T, A, B, C>(
    vertices: A,
    faces: B,
    center: C,
    query: C,
    parallel: bool,
) -> Result<bool, ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    B: AsArray<'a, usize, Ix2>,
    C: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let vertices: ArrayBase<ViewRepr<&'a T>, Ix2> = vertices.into();
    let faces: ArrayBase<ViewRepr<&'a usize>, Ix2> = faces.into();
    let center: ArrayBase<ViewRepr<&'a T>, Ix1> = center.into();
    let query: ArrayBase<ViewRepr<&'a T>, Ix1> = query.into();
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
    if center.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "center",
        });
    }
    if center.len() != 3 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "center",
            expected: 3,
            got: center.len(),
        });
    }
    if query.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "query",
        });
    }
    if query.len() != 3 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "query",
            expected: 3,
            got: query.len(),
        });
    }
    let check_tetrahedrons = |i: usize| {
        let [a_idx, b_idx, c_idx] = [faces[[i, 0]], faces[[i, 1]], faces[[i, 2]]];
        let a = vertices.row(a_idx);
        let b = vertices.row(b_idx);
        let c = vertices.row(c_idx);
        // SAFE: this unwrap is safe because we validated the inputs already
        inside_tetrahedron(a, b, c, center, query).unwrap()
    };
    if parallel {
        Ok((0..faces.dim().0).into_par_iter().any(check_tetrahedrons))
    } else {
        Ok((0..faces.dim().0).any(check_tetrahedrons))
    }
}

/// Determine if a query point is inside a tetrahedron.
///
/// # Description
///
/// Determines if a 3D query point is inside the given tetrahedron's interior.
/// The query point is considered inside the tetrahedron if the point is found
/// in the interior halfspace of each face. The function expects points and
/// vertices in `(pln, row, col)` order.
///
/// # Arguments
///
/// * `a`: Vertex `a` of the oriented plane.
/// * `b`: Vertex `b` of the oriented plane.
/// * `c`: Vertex `c` of the oriented plane.
/// * `d`: The reference point relative to plane `(a, b, c)`.
/// * `query`: The query point to check if inside the polyhedron.
///
/// # Returns
///
/// * `Ok(bool)`: Returns `true` if `query` is inside the tetrahedron, otherwise
///   it returns `false`.
/// * `Err(ImgalError)`: If points `a`, `b`, `c`, `d`, and `query` are empty or
///   do not have length equal to `3`.
pub fn inside_tetrahedron<'a, T, A>(a: A, b: A, c: A, d: A, query: A) -> Result<bool, ImgalError>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let a: ArrayBase<ViewRepr<&'a T>, Ix1> = a.into();
    let b: ArrayBase<ViewRepr<&'a T>, Ix1> = b.into();
    let c: ArrayBase<ViewRepr<&'a T>, Ix1> = c.into();
    let d: ArrayBase<ViewRepr<&'a T>, Ix1> = d.into();
    let query: ArrayBase<ViewRepr<&'a T>, Ix1> = query.into();
    let orient_abc = orient_pred_3d(a, b, c, query)?.is_sign_negative();
    let orient_dba = orient_pred_3d(d, b, a, query)?.is_sign_negative();
    let orient_dcb = orient_pred_3d(d, c, b, query)?.is_sign_negative();
    let orient_dac = orient_pred_3d(d, a, c, query)?.is_sign_negative();
    Ok(orient_abc && orient_dba && orient_dcb && orient_dac)
}

/// Compute the 2D orientation predicate of a triangle.
///
/// # Description
///
/// Computes the 2D orientation predicate of triangle `(o, a, b)` in
/// `(row, col)` dimension order. The sign of the returned value indicates
/// on which side of the oriented line `(o, a)` the point `b` lies:
///
/// - *Positive*: Point `b` lies to the left of the directed line `o -> a`
///   (counterclockwise turn).
/// - *Negative*: Point `b` lies to the right of the directed line `o -> a`
///   (clockwise turn).
/// - *Zero*: Points `o`, `a`, and `b` are collinear.
///
/// # Arguments
///
/// * `o`: The origin vertex of the directed line.
/// * `a`: The endpoint vertex of the directed line.
/// * `b`: The reference point relative to line `(o, a)`.
///
/// # Returns
///
/// * `Ok(f64)`: The orientation of the triangle.
/// * `Err(ImgalError)`: If points `o`, `a`, or `b` are empty or do not have
///   length equal to `3`.
///
/// # Reference
///
/// <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>\
/// <https://doi.org/10.1007/PL00009321>
pub fn orient_pred_2d<'a, T, A>(o: A, a: A, b: A) -> Result<f64, ImgalError>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let o: ArrayBase<ViewRepr<&'a T>, Ix1> = o.into();
    let a: ArrayBase<ViewRepr<&'a T>, Ix1> = a.into();
    let b: ArrayBase<ViewRepr<&'a T>, Ix1> = b.into();
    if o.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray { param_name: "o" });
    }
    if o.len() != 2 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "o",
            expected: 2,
            got: o.len(),
        });
    }
    if a.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray { param_name: "a" });
    }
    if a.len() != 2 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "a",
            expected: 2,
            got: a.len(),
        });
    }
    if b.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray { param_name: "b" });
    }
    if b.len() != 2 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "b",
            expected: 2,
            got: b.len(),
        });
    }
    let [oy, ox] = [o[0].to_f64(), o[1].to_f64()];
    let [ay, ax] = [a[0].to_f64(), a[1].to_f64()];
    let [by, bx] = [b[0].to_f64(), b[1].to_f64()];
    Ok((ax - ox) * (by - oy) - (ay - oy) * (bx - ox))
}

/// Compute the 3D orientation predicate of a tetrahedron.
///
/// # Description
///
/// Computes the 3D orientation prpedicate of tetrahedrion `(a, b, c, d)` in
/// `(pln, row, col)` dimension order. The sign of the returned value indicates
/// on which side of the oriented plane `(a, b, c)` the point `d` lies:
///
/// - *Positive*: Point `d` lies above the plane where `(a, b, c)` appears in
///   counterclockwise order.
/// - *Negative*: Point `d` lies below the plane where `(a, b, c)` appears in
///   clockwise order.
/// - *Zero*: Point `d` is coplanar with plane `(a, b, c)`.
///
/// # Arguments
///
/// * `a`: Vertex `a` of the oriented plane.
/// * `b`: Vertex `b` of the oriented plane.
/// * `c`: Vertex `c` of the oriented plane.
/// * `d`: The reference point relative to plane `(a, b, c)`.
///
/// # Returns
///
/// * `Ok(f64)`: The orientation of the tetrahedron.
/// * `Err(ImgalError)`: If points `a`, `b`, `c` and `d` are empty or do not
///   have length equal to `3`.
///
/// # Reference
///
/// <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>\
/// <https://doi.org/10.1007/PL00009321>
#[inline]
pub fn orient_pred_3d<'a, T, A>(a: A, b: A, c: A, d: A) -> Result<f64, ImgalError>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let a: ArrayBase<ViewRepr<&'a T>, Ix1> = a.into();
    let b: ArrayBase<ViewRepr<&'a T>, Ix1> = b.into();
    let c: ArrayBase<ViewRepr<&'a T>, Ix1> = c.into();
    let d: ArrayBase<ViewRepr<&'a T>, Ix1> = d.into();
    if a.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray { param_name: "a" });
    }
    if a.len() != 3 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "a",
            expected: 3,
            got: a.len(),
        });
    }
    if b.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray { param_name: "b" });
    }
    if b.len() != 3 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "b",
            expected: 3,
            got: b.len(),
        });
    }
    if c.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray { param_name: "c" });
    }
    if c.len() != 3 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "c",
            expected: 3,
            got: c.len(),
        });
    }
    if d.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray { param_name: "d" });
    }
    if d.len() != 3 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "d",
            expected: 3,
            got: d.len(),
        });
    }
    let [az, ay, ax] = [a[0].to_f64(), a[1].to_f64(), a[2].to_f64()];
    let [bz, by, bx] = [b[0].to_f64(), b[1].to_f64(), b[2].to_f64()];
    let [cz, cy, cx] = [c[0].to_f64(), c[1].to_f64(), c[2].to_f64()];
    let [dz, dy, dx] = [d[0].to_f64(), d[1].to_f64(), d[2].to_f64()];
    let [adx, ady, adz] = [ax - dx, ay - dy, az - dz];
    let [bdx, bdy, bdz] = [bx - dx, by - dy, bz - dz];
    let [cdx, cdy, cdz] = [cx - dx, cy - dy, cz - dz];
    Ok(
        adx * (bdy * cdz - bdz * cdy) - ady * (bdx * cdz - bdz * cdx)
            + adz * (bdx * cdy - bdy * cdx),
    )
}

/// Compute the volume of a polyhedron.
///
/// # Description
///
/// Computes the volume of a closed polyhedron defined by `vertices` and
/// `faces`. Each face is turned into a tetrahedron with the `apex` point and
/// their signed volumes summed. The function expects the polyhedron (*i.e.*
/// hull) to have outward-facing normals. This function expects points in
/// `(pln, row, col)` order.
///
/// # Arguments
///
/// * `vertices`: The polyhedron (hull) vertices with `(n_points, 3)` shape.
/// * `faces`: The polyhedron (hull) faces with `(n_triangle, 3)` shape.
/// * `apex`: The shared apex point of all tetrahedra. If `None`, then
///   `[0, 0, 0]` is used. Using a vertex of the hull can improve floating-point
///   accuracy if the hull is far from the origin.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Ok(f64)`: The volume of the polyhedron.
/// * `Err(ImgalError)`: If `vertices` and/or `faces` is empty. If `vertices`
///   and/or `faces` axis 1 `!= 3`.
#[inline]
pub fn polyhedron_volume<'a, T, A, B, C>(
    vertices: A,
    faces: B,
    apex: Option<C>,
    parallel: bool,
) -> Result<f64, ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    B: AsArray<'a, usize, Ix2>,
    C: AsArray<'a, T, Ix1>,
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
    let apex = match apex {
        Some(ap) => ap.into().to_owned(),
        None => Array1::from_vec(vec![T::default(); 3]),
    };
    let polyhedron_vol_calc = |acc: f64, i: usize| {
        let [a_idx, b_idx, c_idx] = [faces[[i, 0]], faces[[i, 1]], faces[[i, 2]]];
        // SAFE: this unwrap is safe because we validated the inputs already
        acc + tetrahedron_volume(
            vertices.row(a_idx),
            vertices.row(b_idx),
            vertices.row(c_idx),
            apex.view(),
        )
        .unwrap()
    };
    if parallel {
        Ok((0..faces.dim().0)
            .into_par_iter()
            .fold(|| 0.0_f64, polyhedron_vol_calc)
            .reduce(|| 0.0_f64, |a, b| a + b)
            .abs())
    } else {
        Ok((0..faces.dim().0).fold(0.0_f64, polyhedron_vol_calc).abs())
    }
}

/// Compute the signed volume of a tetrahedron.
///
/// # Description
///
/// Computes the signed volume of tetrahedron `(a, b, c, d)` in
/// `(pln, row, col)` dimension order. The sign of the returned value indicates
/// on which side of the oriented plane `(a, b, c)` the point `d` lies:
///
/// - *Positive*: Point `d` lies above the plane where `(a, b, c)` appears in
///   counterclockwise order.
/// - *Negative*: Point `d` lies below the plane where `(a, b, c)` appears in
///   clockwise order.
/// - *Zero*: Point `d` is coplanar with plane `(a, b, c)`.
///
/// # Arguments
///
/// * `a`: Vertex `a` of the oriented plane.
/// * `b`: Vertex `b` of the oriented plane.
/// * `c`: Vertex `c` of the oriented plane.
/// * `d`: The reference point relative to plane `(a, b, c)`.
///
/// # Returns
///
/// * `Ok(f64)`: The signed volume of the tetrahedron. Negative signs have
///   volumes pointing towards `d` and positive signs have volumes pointing
///   away.
/// * `Err(ImgalError)`: If points `a`, `b`, `c` and `d` are empty or do not
///   have length equal to `3`.
///
/// # Reference
///
/// <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>\
/// <https://doi.org/10.1007/PL00009321>
#[inline]
pub fn tetrahedron_volume<'a, T, A>(a: A, b: A, c: A, d: A) -> Result<f64, ImgalError>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    Ok(orient_pred_3d(a, b, c, d)? / 6.0)
}
