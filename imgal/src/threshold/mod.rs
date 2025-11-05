//! Threshold functions.
pub mod global;
pub use global::otsu_mask;
pub use global::otsu_value;
pub mod manual;
pub use manual::manual_mask;
