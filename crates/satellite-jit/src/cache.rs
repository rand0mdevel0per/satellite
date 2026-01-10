//! Code hash caching for compiled ABI-OPs.

use satellite_base::{Error, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Cache for compiled code.
pub struct CodeCache {
    /// In-memory cache of code hashes.
    memory_cache: RwLock<HashMap<String, CacheEntry>>,
    /// Bloom filter for fast negative lookups.
    bloom_filter: RwLock<BloomFilter>,
    /// Cache directory.
    cache_dir: PathBuf,
}

/// An entry in the code cache.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// SHA256 hash of the source code.
    pub code_hash: String,
    /// Path to the compiled artifact.
    pub artifact_path: PathBuf,
    /// Whether a truth table is available.
    pub has_truth_table: bool,
}

/// Simple bloom filter for fast lookups.
struct BloomFilter {
    /// Bit array.
    bits: Vec<u64>,
    /// Number of hash functions.
    num_hashes: usize,
}

impl BloomFilter {
    fn new(size: usize) -> Self {
        Self {
            bits: vec![0; size / 64 + 1],
            num_hashes: 3,
        }
    }

    fn insert(&mut self, hash: &str) {
        for i in 0..self.num_hashes {
            let idx = self.hash_fn(hash, i);
            let word = idx / 64;
            let bit = idx % 64;
            if word < self.bits.len() {
                self.bits[word] |= 1 << bit;
            }
        }
    }

    fn may_contain(&self, hash: &str) -> bool {
        for i in 0..self.num_hashes {
            let idx = self.hash_fn(hash, i);
            let word = idx / 64;
            let bit = idx % 64;
            if word >= self.bits.len() || (self.bits[word] & (1 << bit)) == 0 {
                return false;
            }
        }
        true
    }

    fn hash_fn(&self, s: &str, seed: usize) -> usize {
        let mut h = seed as u64;
        for b in s.bytes() {
            h = h.wrapping_mul(31).wrapping_add(b as u64);
        }
        h as usize % (self.bits.len() * 64)
    }
}

impl CodeCache {
    /// Creates a new code cache.
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&cache_dir)?;

        Ok(Self {
            memory_cache: RwLock::new(HashMap::new()),
            bloom_filter: RwLock::new(BloomFilter::new(10000)),
            cache_dir,
        })
    }

    /// Computes the SHA256 hash of code content.
    pub fn hash_code(content: &str) -> String {
        // Simple hash for now, should use SHA256
        let mut h: u64 = 0;
        for b in content.bytes() {
            h = h.wrapping_mul(31).wrapping_add(b as u64);
        }
        format!("{:016x}", h)
    }

    /// Checks if a code hash might be cached (fast check).
    pub fn may_be_cached(&self, code_hash: &str) -> bool {
        self.bloom_filter.read().unwrap().may_contain(code_hash)
    }

    /// Gets a cached entry.
    pub fn get(&self, code_hash: &str) -> Option<CacheEntry> {
        // Fast bloom filter check
        if !self.may_be_cached(code_hash) {
            return None;
        }

        // Check memory cache
        if let Some(entry) = self.memory_cache.read().unwrap().get(code_hash) {
            return Some(entry.clone());
        }

        // Check disk
        self.load_from_disk(code_hash)
    }

    /// Puts an entry in the cache.
    pub fn put(&self, entry: CacheEntry) -> Result<()> {
        let code_hash = entry.code_hash.clone();

        // Update bloom filter
        self.bloom_filter.write().unwrap().insert(&code_hash);

        // Update memory cache
        self.memory_cache.write().unwrap().insert(code_hash, entry);

        Ok(())
    }

    fn load_from_disk(&self, code_hash: &str) -> Option<CacheEntry> {
        let artifact_path = self.cache_dir.join(format!("{}.so", code_hash));
        if artifact_path.exists() {
            let has_truth_table = self
                .cache_dir
                .join(format!("{}_truth_table.bin", code_hash))
                .exists();

            Some(CacheEntry {
                code_hash: code_hash.to_string(),
                artifact_path,
                has_truth_table,
            })
        } else {
            None
        }
    }

    /// Returns the cache directory.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}
