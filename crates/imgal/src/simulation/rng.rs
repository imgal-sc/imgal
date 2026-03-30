use std::ops::Range;

use crate::error::ImgalError;

const INCREMENT: u64 = 1442695040888963407;
const MULTIPLIER: u64 = 6364136223846793005;

/// A 32-bit state Permuted Congruential Generator (PCG) pseudo-random number
/// generator.
///
/// The PCG stores a 64-bit internal state and uses an xorshift and bit rotation
/// to produce u32 values. The generator is deterministic when given the same
/// seed.
#[derive(Clone)]
pub struct Pcg {
    /// The PCG state.
    state: u64,
}

impl Pcg {
    /// Fork the PCG with a random seed value.
    ///
    /// # Description
    ///
    /// Forks the current PCG with a random seed value, returning a new PCG.
    ///
    /// # Returns
    ///
    /// * `Pcg`: A new randomly seeded PCG.
    pub fn fork(&mut self) -> Self {
        Self::new(self.next_u32() as u64)
    }

    /// Create a new seeded PCG.
    ///
    /// # Description
    ///
    /// Creates a new PCG with the given `seed` value.
    ///
    /// # Arguments
    ///
    /// * `seed`: The seed value for the PCG.
    ///
    /// # Returns
    ///
    /// * `Pcg`: A seeded PCG.
    pub fn new(seed: u64) -> Self {
        let mut rng = Self { state: 0 };
        // adding the increment and seed gives the PCG something to work with,
        // this "warms" up the RNG
        rng.state = rng.state.wrapping_add(INCREMENT);
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Return a pseudo-random f32 value.
    ///
    /// # Description
    ///
    /// Returns a pseudo-random f32 value in the half-open interval [0, 1).
    ///
    /// # Returns
    ///
    /// * `f32`: A pseudo-random f32 value in the half-open interval [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Return a pseudo-random u32 value.
    ///
    /// # Description
    ///
    /// Returns a pseudo-random u32 value.
    ///
    /// # Returns
    ///
    /// * `u32`: A pseudo-random u32 value.
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
        let xor_shifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xor_shifted.rotate_right(rot)
    }

    /// Return a pseudo-random u32 value within a given range.
    ///
    /// # Description
    ///
    /// Returns a pseudo-random u32 value within a given range. The start value
    /// of the range must be smaller than the end value.
    ///
    /// # Arguments
    ///
    /// * `r`: The positive range to limit valid pseudo-random u32 values.
    /// # Returns
    ///
    /// * `Ok(u32)`: A pseudo-random u32 value within the given range, `r`.
    /// * `Err(ImgalError)`: If the range start value is larger than the range
    ///   end.
    pub fn next_u32_range(&mut self, r: Range<u32>) -> Result<u32, ImgalError> {
        if r.start > r.end {
            return Err(ImgalError::InvalidPositiveRange {
                start: r.start as usize,
                end: r.end as usize,
            });
        }
        let diff = r.end - r.start;
        // this threshold value is used to avoid "modulo bias" when 2^32 (i.e. u32)
        // can't be evenly divided by the range diff
        let threshold = diff.wrapping_neg() % diff;
        loop {
            let v = self.next_u32();
            if v >= threshold {
                return Ok(r.start + (v % diff));
            }
        }
    }
}
