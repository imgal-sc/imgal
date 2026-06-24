//! Provides spatial data structures and search functions.

pub mod convex_hull;
pub mod geometry;
pub mod halfspace;
mod kd_tree;
pub mod roi;

pub use kd_tree::KDTree;
