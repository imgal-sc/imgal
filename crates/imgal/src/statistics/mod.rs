//! Statistics functions.

mod correlation;
mod min_max;
mod percentile;
mod sample;
mod sort;
mod sum;

pub use correlation::{pearson, weighted_kendall_tau_b};
pub use min_max::max;
pub use min_max::min;
pub use min_max::min_max;
pub use percentile::linear_percentile;
pub use sample::effective_sample_size;
pub use sort::weighted_merge_sort_mut;
pub use sum::kahan_sum;
pub use sum::sum;
