use ndarray::{Array2, Array3};

use crate::error::ImgalError;

/// Create a 2D square kernel with a circular neighborhood.
///
/// # Description
///
/// Creates a boolean kernel representing a filled circle with the specified
/// radius (*i.e.* the neighborhood). The circle is defined using the Euclidean
/// distance from the center point of the kernel. Points within the radius are
/// set to `true`, while points outside are set to `false`.
///
/// # Arguments
///
/// * `radius`: The radius of the circle in pixels. Must be greather than `0`.
///
/// # Returns
///
/// * `Ok(Array2<bool>)`: A 2D square boolean array with side lengths of
///   `radius * 2 + 1` where `true` values represent points inside or on the
///   circle boundary of the specified radius.
/// * `Err(ImgalError)`: If `radius <= 0`.
pub fn circle_kernel(radius: usize) -> Result<Array2<bool>, ImgalError> {
    if radius == 0 {
        return Err(ImgalError::InvalidParameterValueLess {
            param_name: "radius",
            value: 0,
        });
    }
    let dim = radius * 2 + 1;
    let center = radius as f64;
    let mut kernel = Array2::<bool>::default((dim, dim));
    kernel.indexed_iter_mut().for_each(|((row, col), v)| {
        let x = col as f64 - center;
        let y = row as f64 - center;
        let dist = ((x * x) + (y * y)).sqrt();
        *v = dist <= center;
    });
    Ok(kernel)
}

/// Create a 3D kernel with a spherical neighborhood.
///
/// # Description
///
/// Creates a boolean kernel representing a filled sphere with the specified
/// radius (*i.e* the neighborhood). The sphere is defined using the Euclidean
/// distance from the center point of the kernel. Points within the radius are
/// set to `true`, while jpoints outside are set to `false`.
///
/// # Arguments
///
/// * `radius`: The radius of the sphere in voxels. Must be greater than  `0`.
///
/// # Returns
///
/// * `Ok(Array3<bool>)`: A 3D boolean array with side lengths of
///   `radius * 2 + 1` where `true` values represent points inside or on the
///    sphere boundary of the specified radius.
/// * `Err(ImgalError)`: If `radius <= 0`.
pub fn sphere_kernel(radius: usize) -> Result<Array3<bool>, ImgalError> {
    if radius == 0 {
        return Err(ImgalError::InvalidParameterValueEqual {
            param_name: "radius",
            value: 0,
        });
    }
    // set the sphere parameters and calculate the Euclidean distance at each
    // position
    let dim = radius * 2 + 1;
    let center = radius as f64;
    let mut kernel = Array3::<bool>::default((dim, dim, dim));
    kernel.indexed_iter_mut().for_each(|((pln, row, col), v)| {
        let x = col as f64 - center;
        let y = row as f64 - center;
        let z = pln as f64 - center;
        let dist = ((x * x) + (y * y) + (z * z)).sqrt();
        *v = dist <= center;
    });
    Ok(kernel)
}

/// Create a 2D kernel with a weighted circular neighborhood.
///
/// # Description
///
/// Creates a kernel representing a weighted value circle with the specified
/// radius (*i.e.* the neighborhood). The circle is defined using the Euclidean
/// distance from the center point of the kernel. Points within the radius are
/// valid weighted positions (*i.e.* a weight can be assigned but is not
/// guaranteed to be present), while points outside are not valid and set to
/// `0.0`. The maximum weight value is located at the center of the circle,
/// defined by `initial_value`, and decaying values towards the edge at the
/// `falloff_radius` rate.
///
/// # Arguments
///
/// * `circle_radius`: The radius of the circle in pixels. Must be greater than
///   `0`.
/// * `falloff_radius`: A scaling factor that determines how quickly weights
///   decay with distance. Larger values result in a slower falloff with a
///   broader circle. Small values result in a faster falloff with a tighter
///   circle.
/// * `initial_value`: The maximum weight value at the center of the kernel. If
///   `None`, then `initial_value = 1.0`
///
/// # Returns
///
/// * `Ok(Array2<f64>)`: A 2D array with side lengths of `radius * 2 + 1` with a
///   weighted circular neighborhood.
/// * `Err(ImgalError)`: If circle `radius <= 0`.
pub fn weighted_circle_kernel(
    circle_radius: usize,
    falloff_radius: f64,
    initial_value: Option<f64>,
) -> Result<Array2<f64>, ImgalError> {
    if circle_radius == 0 {
        return Err(ImgalError::InvalidParameterValueLess {
            param_name: "circle_radius",
            value: 0,
        });
    }
    // set the circle parameters and calculate the Euclidean distance at each
    // position with weights values decreasing towards the edge defined by the
    // "falloff radius"
    let dim = circle_radius * 2 + 1;
    let center = circle_radius as f64;
    let norm_center = center / falloff_radius;
    let iv = initial_value.unwrap_or(1.0);
    let mut kernel = Array2::<f64>::zeros((dim, dim));
    kernel.indexed_iter_mut().for_each(|((row, col), v)| {
        let x = col as f64 - center;
        let y = row as f64 - center;
        let mut norm_dist = ((x * x) + (y * y)).sqrt() / falloff_radius;
        if norm_dist <= norm_center {
            if norm_dist >= iv {
                norm_dist = 0.0;
            } else {
                norm_dist = iv - norm_dist;
            }
            *v = norm_dist;
        } else {
            *v = 0.0;
        }
    });
    Ok(kernel)
}

/// Create a 3D kernel with a weighted spherical neighborhood.
///
/// # Description
///
/// Creates a kernel representing a weighted value sphere with the specified
/// radius (*i.e.* the neighborhood). The sphere is defined using the Euclidean
/// distance from the center point of the kernel. Points within the radius are
/// valid weighted positions (*i.e.* a weight can be assigned but is not
/// guaranteed to be present), while points outside are not valid and set to
/// `0.0`. The maximum weight value is located at the center of the sphere,
/// defined by `initial_value`, and decaying values towards the edge at the
/// `falloff_radius` rate.
///
/// # Arguments
///
/// * `sphere_radius`: The radius of the sphere in voxels. Must be greater than
///   `0`.
/// * `falloff_radius`: A scaling factor that determines how quickly weights
///   decay with distance. Larger values result in a slower falloff with a
///   broader sphere. Small values result in a faster falloff with a tighter
///   sphere.
/// * `initial_value`: The maximum weight value at the center of the kernel. If
///   `None` then `initial_value = 1.0`.
///
/// # Returns
///
/// * `OK(Array3<f64>)`: A 3D array with side lengths of `radius * 2 + 1` with a
///   weighted spherical neighborhood.
/// * `Err(ImgalError)`: If sphere `radius <= 0`.
pub fn weighted_sphere_kernel(
    sphere_radius: usize,
    falloff_radius: f64,
    initial_value: Option<f64>,
) -> Result<Array3<f64>, ImgalError> {
    if sphere_radius == 0 {
        return Err(ImgalError::InvalidParameterValueLess {
            param_name: "sphere_radius",
            value: 0,
        });
    }
    // set the sphere parameters and calculate the Euclidean distance at each
    // position with weights values decreasing towards the edge defined by the
    // "falloff radius"
    let dim = sphere_radius * 2 + 1;
    let center = sphere_radius as f64;
    let norm_center = center / falloff_radius;
    let iv = initial_value.unwrap_or(1.0);
    let mut kernel = Array3::<f64>::zeros((dim, dim, dim));
    kernel.indexed_iter_mut().for_each(|((pln, row, col), v)| {
        let x = col as f64 - center;
        let y = row as f64 - center;
        let z = pln as f64 - center;
        let mut norm_dist = ((x * x) + (y * y) + (z * z)).sqrt() / falloff_radius;
        if norm_dist <= norm_center {
            if norm_dist >= iv {
                norm_dist = 0.0;
            } else {
                norm_dist = iv - norm_dist;
            }
            *v = norm_dist;
        } else {
            *v = 0.0;
        }
    });
    Ok(kernel)
}
