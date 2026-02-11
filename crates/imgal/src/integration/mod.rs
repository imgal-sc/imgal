//! Numerical integration functions.
pub mod rectangle;
pub use rectangle::midpoint;

pub mod simpson;
pub use simpson::composite_simpson;
pub use simpson::simpson;
