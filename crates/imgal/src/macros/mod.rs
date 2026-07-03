/// Dispatch work to either a sequential iterator or parallel execution path
/// with the requested thread count.
///
/// # Description
///
/// This macro dispatches work to either a sequential or parallel execution path
/// with the requested, max or auto-detected max thread counts.
///
/// # Arguments
///
/// * `$threads`: An `Option<usize>` thread count. `None` or `Some(1)` runs
///   sequentially. `Some(0)` uses all available threads. `Some(n)` uses `n`
///   threads, clamped to the number of available logical CPUs.
///
///
/// # Example
///
/// ```ignore
/// let threads = Some(2);
/// let data: Vec<usize> = par!(
///      threads,
///      seq_exp: (0..10).map(|i| i * 2).collect(),
///      par_exp: (0..10).into_par_iter().map(|i| i * 2).collect()
///  );
/// ```
macro_rules! par {
    ($threads:expr, seq_exp: $seq:expr, par_exp: $par:expr) => {{
        match $threads.unwrap_or(1) {
            1 => $seq,
            0 => $par,
            n => crate::macros::get_pool(n).install(|| $par),
        }
    }};
}

/// Helper function to construct thread pools for the par! macro.
pub fn get_pool(n: usize) -> rayon::ThreadPool {
    let max = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n.min(max))
        .build()
        .unwrap()
}
