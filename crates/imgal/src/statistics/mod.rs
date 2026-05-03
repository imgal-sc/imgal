//! Statistics functions.
mod correlation;
pub use correlation::{pearson, weighted_kendall_tau_b};
mod min_max;
pub use min_max::max;
pub use min_max::min;
pub use min_max::min_max;
mod percentile;
pub use percentile::linear_percentile;
mod sample;
pub use sample::effective_sample_size;
mod sum;
pub use sum::kahan_sum;
pub use sum::sum;
mod sort;
pub use sort::weighted_merge_sort_mut;
