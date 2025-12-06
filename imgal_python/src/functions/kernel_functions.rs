use numpy::{IntoPyArray, PyArray2, PyArray3};
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::kernel;

/// Create a 2-dimensional square kernel with a circle neighborhood.
///
/// Creates a square boolean kernel representing a filled circle of the
/// specified radius (_i.e._ the neighborhood). The circle is defined using the
/// Euclidean distance from the center point. Points within the radius are set
/// to `true`, while points outside are set to `false`.
///
/// Args:
///     radius: The radius of the circle in pixels. Must be greather than `0`.
///
/// Returns:
///     A 2-dimensional square boolean array with side lengths of
///     `radius * 2 + 1` where `true` values represent points inside or on the
///     circle boundary of the specified radius.
#[pyfunction]
#[pyo3(name = "circle_kernel")]
pub fn neighborhood_circle_kernel(py: Python, radius: usize) -> PyResult<Bound<PyArray2<bool>>> {
    kernel::neighborhood::circle_kernel(radius)
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
}

/// Create a 3-dimensional cube kernel with a sphere neighborhood.
///
/// Creates a cube boolean kernel representing a filled sphere of the specified
/// radius (_i.e_ the neighborhood). The sphere is defined using the Euclidean
/// distance from the center point. Points within the radius are set to `true`,
/// while jpoints outside are set to `false`.
///
/// Args:
///     radius: The radius of the sphere in voxels. Must be greater than  `0`.
///
/// Returns:
///     A 3-dimensional cube boolean array with side lengths of `radius * 2 + 1`
///     where `true` values represent points inside or on the sphere boundary of
///     the specified radius.
#[pyfunction]
#[pyo3(name = "sphere_kernel")]
pub fn neighborhood_sphere_kernel(py: Python, radius: usize) -> PyResult<Bound<PyArray3<bool>>> {
    kernel::neighborhood::sphere_kernel(radius)
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
}

/// Create a 2-dimensional square kernel with a weighted circle neighborhood.
///
/// Creates a square kernel representing a weighted value circle of the
/// specified radius (_i.e._ the neighborhood). The circle is defined using the
/// Euclidean distance from the center point. Points within the radius are valid
/// weighted positions (_i.e._ a weight can be assigned but is not guaranteed to
/// be present), while points outside are not valid and set to `0.0`. The
/// maximum weight value is located at the center of the circle, defined by
/// `initial_value`, and decaying values towards the edge at the
/// `falloff_radius` rate.
///
/// Args:
///     circle_radius: The radius of the circle in pixels. Must be greater than
///         `0`.
///     falloff_radius: A scaling factor that determines how quickly weights
///         decay with distance. Larger values result in a slower falloff with a
///         broader circle. Small values result in a faster falloff with a
///         tighter circle.
///     initial_value: The maximum weight value at the center of the kernel. If
///         `None`, then `initial_value = 1.0`
///
/// Returns:
///      A 2-dimensional square array with side lengths of `radius * 2 + 1` with
//       a weighted circular neighborhood.
#[pyfunction]
#[pyo3(name = "weighted_circle_kernel")]
#[pyo3(signature = (circle_radius, falloff_radius, initial_value=None))]
pub fn neighborhood_weighted_circle_kernel(
    py: Python,
    circle_radius: usize,
    falloff_radius: f64,
    initial_value: Option<f64>,
) -> PyResult<Bound<PyArray2<f64>>> {
    kernel::neighborhood::weighted_circle_kernel(circle_radius, falloff_radius, initial_value)
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
}

/// Create a 3-dimensional cube kernel with a weighted sphere neighborhood.
///
/// Creates a cube kernel representing a weighted value sphere of the specified
/// radius (_i.e._ the neighborhood). The sphere is defined using the Euclidean
/// distance from the center point. Points within the radius are valid weighted
/// positions (_i.e._ a weight can be assigned but is not guaranteed to be
/// present), while points outside are not valid and set to `0.0`. The maximum
/// weight value is located at the center of the sphere, defined by
/// `initial_value`, and decaying values towards the edge at the
/// `falloff_radius` rate.
///
/// Args:
///     sphere_radius: The radius of the sphere in voxels. Must be greater than
///         `0`.
///     falloff_radius: A scaling factor that determines how quickly weights
///         decay with distance. Larger values result in a slower falloff with a
///         broader sphere. Small values result in a faster falloff with a
///         tighter sphere.
///     initial_value: The maximum weight value at the center of the kernel. If
///         `None` then `initial_value = 1.0`.
///
/// Returns:
///     A 3-dimensional cube array with side lengths of `radius * 2 + 1` with a
///     weighted spherical neighborhood.
#[pyfunction]
#[pyo3(name = "weighted_sphere_kernel")]
#[pyo3(signature = (sphere_radius, falloff_radius, initial_value=None))]
pub fn neighborhood_weighted_sphere_kernel(
    py: Python,
    sphere_radius: usize,
    falloff_radius: f64,
    initial_value: Option<f64>,
) -> PyResult<Bound<PyArray3<f64>>> {
    kernel::neighborhood::weighted_sphere_kernel(sphere_radius, falloff_radius, initial_value)
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
}
