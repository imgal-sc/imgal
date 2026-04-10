//! Numerical integration functions.
mod rectangle;
pub use rectangle::midpoint;
mod simpson;
pub use simpson::composite_simpson;
pub use simpson::simpson;
