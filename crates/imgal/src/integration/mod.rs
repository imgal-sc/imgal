//! Numerical integration functions.

mod rectangle;
mod simpson;

pub use rectangle::midpoint;
pub use simpson::composite_simpson;
pub use simpson::simpson;
