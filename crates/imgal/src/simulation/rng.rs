use std::ops::Range;

const INCREMENT: u64 = 1442695040888963407;
const MULTIPLIER: u64 = 6364136223846793005;

/// A 64-bit state PRNG that produces a 32-bit output fixed with a 64-bit stream
/// selector and 64-bit mulitpplier constant.
pub struct Pcg {
    state: u64,
}

impl Pcg {
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
        };
        // adding in the increment and seed gives the PCG something to work
        // with, this "warms" up the RNG
        rng.state = rng.state.wrapping_add(INCREMENT);
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    pub fn net_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
        let xor_shifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xor_shifted.rotate_right(rot)
    }

    pub fn next_u32_range(&mut self, r: Range<u32>) -> u32 {
        todo!();
    }
}
