//! Numerical integration functions.
pub(crate) mod rectangle;
pub use rectangle::midpoint;
pub(crate) mod simpson;
pub use simpson::composite_simpson;
pub use simpson::simpson;
