//! Convex hull, geometry, halfplane/halfspace and spatial data structures.
//!
//! This module provides various functions for computing 2D and 3D convex hulls,
//! geometric operations, and spatial data structures (*e.g* Kd-Trees and
//! regions of interest (ROIs)).

pub mod convex_hull;
pub mod geometry;
pub mod halfplane;
pub mod halfspace;
mod kd_tree;
pub mod roi;

pub use kd_tree::KdTree;
