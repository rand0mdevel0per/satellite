//! Truth table optimization for small ABI-OPs.

use satellite_base::Result;

/// A precomputed truth table.
#[derive(Debug, Clone)]
pub struct TruthTable {
    /// Number of input bits.
    input_bits: usize,
    /// Number of output bits.
    output_bits: usize,
    /// Table data (output for each input combination).
    data: Vec<u64>,
}

impl TruthTable {
    /// Maximum input bits for truth table optimization.
    pub const MAX_INPUT_BITS: usize = 20;

    /// Creates a new truth table by exhaustive evaluation.
    pub fn generate<F>(input_bits: usize, output_bits: usize, eval_fn: F) -> Result<Self>
    where
        F: Fn(u64) -> u64,
    {
        if input_bits > Self::MAX_INPUT_BITS {
            return Err(satellite_base::Error::InvalidDimension(input_bits));
        }

        let size = 1usize << input_bits;
        let mut data = Vec::with_capacity(size);

        for i in 0..size as u64 {
            data.push(eval_fn(i));
        }

        Ok(Self {
            input_bits,
            output_bits,
            data,
        })
    }

    /// Looks up the output for a given input.
    #[inline]
    pub fn lookup(&self, input: u64) -> u64 {
        self.data[input as usize]
    }

    /// Returns the number of input bits.
    pub fn input_bits(&self) -> usize {
        self.input_bits
    }

    /// Returns the number of output bits.
    pub fn output_bits(&self) -> usize {
        self.output_bits
    }

    /// Returns the table size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<u64>()
    }

    /// Serializes to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16 + self.data.len() * 8);
        bytes.extend_from_slice(&(self.input_bits as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.output_bits as u64).to_le_bytes());
        for &val in &self.data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Deserializes from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 {
            return Err(satellite_base::Error::Serialization(
                "Invalid truth table".to_string(),
            ));
        }

        let input_bits = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let output_bits = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;

        let size = 1usize << input_bits;
        let mut data = Vec::with_capacity(size);

        for i in 0..size {
            let start = 16 + i * 8;
            let val = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
            data.push(val);
        }

        Ok(Self {
            input_bits,
            output_bits,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truth_table() {
        let table = TruthTable::generate(4, 1, |x| x & 1).unwrap();
        assert_eq!(table.lookup(0), 0);
        assert_eq!(table.lookup(1), 1);
        assert_eq!(table.lookup(2), 0);
        assert_eq!(table.lookup(3), 1);
    }
}
