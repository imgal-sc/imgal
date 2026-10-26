use pyo3::prelude::*;

/// Convex hull, geometry, halfplane/halfspace and spatial data structures.
///
/// This module provides various functions for computing 2D and 3D convex hulls,
/// geometric operations, and spatial data structures (*e.g* Kd-Trees and
/// regions of interest (ROIs)).
#[pymodule(submodule)]
pub mod spatial {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::child_modules::spatial_module::convex_hull;
    #[pymodule_export]
    use crate::child_modules::spatial_module::geometry;
    #[pymodule_export]
    use crate::child_modules::spatial_module::halfplane;
    #[pymodule_export]
    use crate::child_modules::spatial_module::halfspace;
    #[pymodule_export]
    use crate::child_modules::spatial_module::roi;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.spatial")
    }
}

/// 2D and 3D convex hull functions.
///
/// This module provides functions for computing 2D and 3D convex hulls.
#[pymodule(submodule)]
pub mod convex_hull {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::spatial_functions::convex_hull_chan_2d;
    #[pymodule_export]
    use crate::functions::spatial_functions::convex_hull_graham_scan;
    #[pymodule_export]
    use crate::functions::spatial_functions::convex_hull_jarvis_march;
    #[pymodule_export]
    use crate::functions::spatial_functions::convex_hull_quickhull_3d;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.spatial.convex_hull")
    }
}

/// Computational geometry functions.
///
/// This module provides fundamental computational geometry functions for
/// working with polygons and polyhedra.
#[pymodule(submodule)]
pub mod geometry {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::spatial_functions::geometry_hull_centroid;
    #[pymodule_export]
    use crate::functions::spatial_functions::geometry_inside_polyhedron;
    #[pymodule_export]
    use crate::functions::spatial_functions::geometry_inside_tetrahedron;
    #[pymodule_export]
    use crate::functions::spatial_functions::geometry_orient_pred_2d;
    #[pymodule_export]
    use crate::functions::spatial_functions::geometry_orient_pred_3d;
    #[pymodule_export]
    use crate::functions::spatial_functions::geometry_polyhedron_volume;
    #[pymodule_export]
    use crate::functions::spatial_functions::geometry_tetrahedron_volume;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.spatial.geometry")
    }
}

/// Halfplane functions.
///
/// This module provides functions for converting 2D convex hulls into
/// halfplanes, intersecting them and querying interior points.
#[pymodule(submodule)]
pub mod halfplane {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::spatial_functions::halfplane_edge_to_halfplane;
    #[pymodule_export]
    use crate::functions::spatial_functions::halfplane_halfplane_intersection;
    #[pymodule_export]
    use crate::functions::spatial_functions::halfplane_hull_to_halfplane;
    #[pymodule_export]
    use crate::functions::spatial_functions::halfplane_inside_halfplane_interior;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.spatial.halfplane")
    }
}

/// Halfspace functions.
///
/// This module provides functions for converting 3D convex hulls into
/// halfplanes, intersecting them and querying interior points.
#[pymodule(submodule)]
pub mod halfspace {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::spatial_functions::halfspace_face_to_halfspace;
    #[pymodule_export]
    use crate::functions::spatial_functions::halfspace_halfspace_intersection;
    #[pymodule_export]
    use crate::functions::spatial_functions::halfspace_hull_to_halfspace;
    #[pymodule_export]
    use crate::functions::spatial_functions::halfspace_inside_halfspace_inerior;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.spatial.halfspace")
    }
}

/// *n*-dimensional region of interest (ROI) functions.
///
/// This module provides functions for creating *n*-dimensional ROIs and
/// querying ROI bounded data.
#[pymodule(submodule)]
pub mod roi {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::spatial_functions::roi_roi_cloud_map;
    #[pymodule_export]
    use crate::functions::spatial_functions::roi_roi_data_map;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.spatial.roi")
    }
}
