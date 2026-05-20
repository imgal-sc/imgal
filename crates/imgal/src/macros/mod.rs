/// Dispatch work to either a sequential iterator or parallel iterator with the
/// requested thread pool count.
macro_rules! par {
    ($threads:expr, seq_exp: $seq:expr, par_exp: $par:expr) => {{
        let threads = $threads.unwrap_or(0);
        let resolve_threads = |req: usize| {
            let max_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            match req {
                0 => max_threads,
                n => n.min(max_threads),
            }
        };
        let get_pool = |n: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap()
        };
        match resolve_threads(threads) {
            1 => $seq,
            n => get_pool(n).install(|| $par),
        }
    }};
}
