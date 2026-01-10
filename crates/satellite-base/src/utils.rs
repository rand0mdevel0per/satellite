//! Utility functions and helpers.

use std::hash::{Hash, Hasher};

/// Fast xorshift PRNG for non-cryptographic randomness.
#[derive(Debug, Clone)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Creates a new PRNG with the given seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    /// Generates the next random u64.
    pub fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generates a random f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next() as f64) / (u64::MAX as f64)
    }
}

/// Computes a fast hash of the given value.
#[must_use]
pub fn fast_hash<T: Hash>(value: &T) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Rounds up to the next power of two.
#[must_use]
pub const fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        1 << (usize::BITS - (n - 1).leading_zeros())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift() {
        let mut rng = XorShift64::new(12345);
        let a = rng.next();
        let b = rng.next();
        assert_ne!(a, b);
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
    }
}
