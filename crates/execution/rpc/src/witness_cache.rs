//! Disk-backed ring of prebuilt `debug_executePayload` execution witnesses.

use std::{
    collections::HashMap,
    fs,
    io::{self, Write},
    num::NonZeroUsize,
    ops::RangeInclusive,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{Arc, Mutex},
};

use alloy_primitives::{B64, B256, Bytes, Keccak256, keccak256};
use alloy_rlp::{Decodable, Encodable};
use alloy_rpc_types_debug::ExecutionWitness;
use alloy_rpc_types_engine::PayloadId;
use base_execution_payload_builder::Attributes;
use tracing::warn;

use crate::metrics::WitnessCacheMetrics;

/// File extension of committed witness cache entries.
const ENTRY_EXTENSION: &str = "witness.zst";

/// File extension of in-progress witness cache writes.
const TEMP_EXTENSION: &str = "tmp";

/// Configuration for the prebuilt witness cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessCacheConfig {
    /// Directory holding one compressed file per cached witness.
    pub path: PathBuf,
    /// Number of blocks behind the proofs tip to keep cached witnesses for.
    pub retention_blocks: u64,
    /// Minimum number of blocks a block must be behind the proofs tip before it is prebuilt.
    pub build_lag: u64,
    /// Maximum number of witnesses built concurrently in the background.
    pub builder_concurrency: NonZeroUsize,
}

/// Lookup key of a cached witness: the payload job a `debug_executePayload` request maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WitnessCacheKey {
    /// Hash of the parent block the payload is built on.
    pub parent_hash: B256,
    /// Payload ID derived from the payload attributes.
    pub payload_id: PayloadId,
}

impl WitnessCacheKey {
    /// Returns the key and transaction list hash of a payload built from `attributes`.
    pub fn from_attributes<A: Attributes>(parent_hash: B256, attributes: &A) -> (Self, B256) {
        let transactions_hash = WitnessCache::transactions_hash(
            attributes.sequencer_transactions().iter().map(|tx| tx.encoded_bytes()),
        );
        (Self { parent_hash, payload_id: attributes.payload_job_id() }, transactions_hash)
    }
}

/// Index metadata of a cached witness file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessCacheEntry {
    /// Number of the block the witness was built for.
    pub block_number: u64,
    /// Hash of the payload transaction list, see [`WitnessCache::transactions_hash`].
    ///
    /// Verified on lookup because the payload ID is only 8 bytes.
    pub transactions_hash: B256,
    /// Size of the compressed witness file in bytes.
    pub size: u64,
}

impl WitnessCacheEntry {
    /// Returns the file name encoding this entry and its key.
    pub fn file_name(&self, key: &WitnessCacheKey) -> String {
        format!(
            "{:020}-{:x}-{:x}-{:x}.{ENTRY_EXTENSION}",
            self.block_number, key.parent_hash, key.payload_id.0, self.transactions_hash
        )
    }

    /// Parses a file name produced by [`Self::file_name`]; `size` is left as zero.
    pub fn parse_file_name(name: &str) -> Option<(WitnessCacheKey, Self)> {
        let stem = name.strip_suffix(ENTRY_EXTENSION)?.strip_suffix('.')?;
        let mut parts = stem.split('-');
        let block_number = parts.next()?.parse().ok()?;
        let parent_hash = B256::from_str(parts.next()?).ok()?;
        let payload_id = PayloadId(B64::from_str(parts.next()?).ok()?);
        let transactions_hash = B256::from_str(parts.next()?).ok()?;
        if parts.next().is_some() {
            return None;
        }
        Some((
            WitnessCacheKey { parent_hash, payload_id },
            Self { block_number, transactions_hash, size: 0 },
        ))
    }
}

/// Bounded on-disk ring of execution witnesses, one zstd-compressed file per witness.
///
/// The in-memory index is rebuilt from the directory on [`Self::open`], so entries survive
/// restarts. Writes are atomic (temp file + rename). Entries are removed by
/// [`Self::evict_before`].
#[derive(Debug)]
pub struct WitnessCache {
    dir: PathBuf,
    entries: Mutex<HashMap<WitnessCacheKey, WitnessCacheEntry>>,
}

impl WitnessCache {
    /// Opens the cache at `dir`, creating it if needed and indexing existing entries.
    ///
    /// Leftover temporary files from interrupted writes are removed.
    pub fn open(dir: impl Into<PathBuf>) -> io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;

        let mut entries = HashMap::new();
        for dir_entry in fs::read_dir(&dir)? {
            let dir_entry = dir_entry?;
            let path = dir_entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else { continue };
            if path.extension().is_some_and(|ext| ext == TEMP_EXTENSION) {
                fs::remove_file(&path)?;
                continue;
            }
            if let Some((key, mut entry)) = WitnessCacheEntry::parse_file_name(name) {
                entry.size = dir_entry.metadata()?.len();
                entries.insert(key, entry);
            }
        }

        Self::record_size_metrics(&entries);
        Ok(Self { dir, entries: Mutex::new(entries) })
    }

    /// Hashes a payload transaction list given the EIP-2718 encoding of each transaction.
    pub fn transactions_hash<'a>(transactions: impl IntoIterator<Item = &'a Bytes>) -> B256 {
        let mut hasher = Keccak256::new();
        for tx in transactions {
            hasher.update(keccak256(tx));
        }
        hasher.finalize()
    }

    /// Returns whether a witness is cached for `key`.
    pub fn contains(&self, key: &WitnessCacheKey) -> bool {
        self.entries.lock().expect("witness cache lock poisoned").contains_key(key)
    }

    /// Returns the inclusive block range of cached witnesses.
    pub fn block_range(&self) -> Option<RangeInclusive<u64>> {
        let entries = self.entries.lock().expect("witness cache lock poisoned");
        let min = entries.values().map(|entry| entry.block_number).min()?;
        let max = entries.values().map(|entry| entry.block_number).max()?;
        Some(min..=max)
    }

    /// Returns the cached witness for `key` if its transaction list hash matches.
    ///
    /// Unreadable entries are dropped from the index and treated as misses.
    pub fn get(&self, key: &WitnessCacheKey, transactions_hash: B256) -> Option<ExecutionWitness> {
        let entry = *self.entries.lock().expect("witness cache lock poisoned").get(key)?;
        if entry.transactions_hash != transactions_hash {
            return None;
        }

        let path = self.dir.join(entry.file_name(key));
        match fs::read(&path).and_then(|data| Self::decode_witness(&data)) {
            Ok(witness) => Some(witness),
            Err(error) => {
                warn!(error = %error, path = %path.display(), "dropping unreadable witness cache entry");
                self.remove(key, &entry);
                None
            }
        }
    }

    /// Returns the cached witness for `key`, or the result of `build` on a miss.
    ///
    /// A built witness is cached if `block_number` lies within the cached block range, so
    /// requests for blocks the background builder skipped still populate the ring.
    pub async fn get_or_build<E>(
        self: &Arc<Self>,
        block_number: u64,
        key: WitnessCacheKey,
        transactions_hash: B256,
        build: impl Future<Output = Result<ExecutionWitness, E>>,
    ) -> Result<ExecutionWitness, E> {
        let cache = Arc::clone(self);
        match tokio::task::spawn_blocking(move || cache.get(&key, transactions_hash)).await {
            Ok(Some(witness)) => {
                WitnessCacheMetrics::hits().increment(1);
                return Ok(witness);
            }
            Ok(None) => {}
            Err(error) => warn!(error = %error, "witness cache lookup failed"),
        }
        WitnessCacheMetrics::misses().increment(1);

        let witness = build.await?;
        if self.block_range().is_some_and(|range| range.contains(&block_number)) {
            let cache = Arc::clone(self);
            let cached = witness.clone();
            let inserted = tokio::task::spawn_blocking(move || {
                cache.insert(block_number, key, transactions_hash, &cached)
            })
            .await;
            match inserted {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    warn!(error = %error, block = block_number, "failed to cache witness")
                }
                Err(error) => {
                    warn!(error = %error, block = block_number, "failed to cache witness")
                }
            }
        }
        Ok(witness)
    }

    /// Atomically writes `witness` to disk and indexes it under `key`.
    pub fn insert(
        &self,
        block_number: u64,
        key: WitnessCacheKey,
        transactions_hash: B256,
        witness: &ExecutionWitness,
    ) -> io::Result<()> {
        let data = Self::encode_witness(witness)?;
        let entry = WitnessCacheEntry { block_number, transactions_hash, size: data.len() as u64 };
        let path = self.dir.join(entry.file_name(&key));
        let temp_path = path.with_extension(TEMP_EXTENSION);

        let mut file = fs::File::create(&temp_path)?;
        file.write_all(&data)?;
        file.sync_data()?;
        fs::rename(&temp_path, &path)?;

        let mut entries = self.entries.lock().expect("witness cache lock poisoned");
        if let Some(previous) = entries.insert(key, entry)
            && previous.file_name(&key) != entry.file_name(&key)
        {
            Self::remove_file(&self.dir.join(previous.file_name(&key)));
        }
        Self::record_size_metrics(&entries);
        Ok(())
    }

    /// Removes all entries for blocks below `block_number`.
    pub fn evict_before(&self, block_number: u64) {
        let mut entries = self.entries.lock().expect("witness cache lock poisoned");
        entries.retain(|key, entry| {
            let keep = entry.block_number >= block_number;
            if !keep {
                Self::remove_file(&self.dir.join(entry.file_name(key)));
            }
            keep
        });
        Self::record_size_metrics(&entries);
    }

    /// Serializes and compresses a witness into the on-disk format.
    pub fn encode_witness(witness: &ExecutionWitness) -> io::Result<Vec<u8>> {
        let mut rlp = Vec::new();
        witness.state.encode(&mut rlp);
        witness.codes.encode(&mut rlp);
        witness.keys.encode(&mut rlp);
        witness.headers.encode(&mut rlp);
        zstd::encode_all(rlp.as_slice(), zstd::DEFAULT_COMPRESSION_LEVEL)
    }

    /// Decompresses and deserializes a witness from the on-disk format.
    pub fn decode_witness(data: &[u8]) -> io::Result<ExecutionWitness> {
        let rlp = zstd::decode_all(data)?;
        let buf = &mut rlp.as_slice();
        let decode = |buf: &mut &[u8]| {
            Vec::<Bytes>::decode(buf)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
        };
        let witness = ExecutionWitness {
            state: decode(buf)?,
            codes: decode(buf)?,
            keys: decode(buf)?,
            headers: decode(buf)?,
        };
        if !buf.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "trailing witness bytes"));
        }
        Ok(witness)
    }

    fn remove(&self, key: &WitnessCacheKey, entry: &WitnessCacheEntry) {
        let mut entries = self.entries.lock().expect("witness cache lock poisoned");
        if entries.get(key) == Some(entry) {
            entries.remove(key);
            Self::remove_file(&self.dir.join(entry.file_name(key)));
            Self::record_size_metrics(&entries);
        }
    }

    fn remove_file(path: &Path) {
        if let Err(error) = fs::remove_file(path)
            && error.kind() != io::ErrorKind::NotFound
        {
            warn!(error = %error, path = %path.display(), "failed to remove witness cache entry");
        }
    }

    fn record_size_metrics(entries: &HashMap<WitnessCacheKey, WitnessCacheEntry>) {
        WitnessCacheMetrics::entries().set(entries.len() as f64);
        WitnessCacheMetrics::bytes()
            .set(entries.values().map(|entry| entry.size).sum::<u64>() as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn witness(seed: u8) -> ExecutionWitness {
        ExecutionWitness {
            state: vec![Bytes::from(vec![seed; 64]), Bytes::from(vec![seed + 1; 3])],
            codes: vec![Bytes::from(vec![seed; 100])],
            keys: vec![],
            headers: vec![Bytes::from(vec![seed; 500])],
        }
    }

    fn key(seed: u8) -> WitnessCacheKey {
        WitnessCacheKey {
            parent_hash: B256::repeat_byte(seed),
            payload_id: PayloadId::new([seed; 8]),
        }
    }

    fn files(dir: &Path) -> Vec<String> {
        let mut names: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn insert_get_roundtrip_verifies_transactions_hash() {
        let dir = tempfile::tempdir().unwrap();
        let cache = WitnessCache::open(dir.path()).unwrap();
        let txs_hash = WitnessCache::transactions_hash(&[Bytes::from_static(b"tx")]);

        cache.insert(10, key(1), txs_hash, &witness(1)).unwrap();

        assert!(cache.contains(&key(1)));
        assert_eq!(cache.get(&key(1), txs_hash), Some(witness(1)));
        assert_eq!(cache.get(&key(1), B256::ZERO), None);
        assert_eq!(cache.get(&key(2), txs_hash), None);
        assert_eq!(cache.block_range(), Some(10..=10));
    }

    #[test]
    fn evict_before_removes_old_entries_and_files() {
        let dir = tempfile::tempdir().unwrap();
        let cache = WitnessCache::open(dir.path()).unwrap();
        for block in 1..=5u8 {
            cache.insert(block.into(), key(block), B256::ZERO, &witness(block)).unwrap();
        }

        cache.evict_before(4);

        assert_eq!(cache.block_range(), Some(4..=5));
        assert!(!cache.contains(&key(3)));
        assert_eq!(cache.get(&key(4), B256::ZERO), Some(witness(4)));
        assert_eq!(files(dir.path()).len(), 2);
    }

    #[test]
    fn reopen_rebuilds_index_and_removes_temp_files() {
        let dir = tempfile::tempdir().unwrap();
        let txs_hash = B256::repeat_byte(7);
        {
            let cache = WitnessCache::open(dir.path()).unwrap();
            cache.insert(42, key(1), txs_hash, &witness(1)).unwrap();
            cache.insert(43, key(2), txs_hash, &witness(2)).unwrap();
        }
        fs::write(dir.path().join("interrupted.tmp"), b"partial").unwrap();
        fs::write(dir.path().join("unrelated.txt"), b"ignored").unwrap();

        let cache = WitnessCache::open(dir.path()).unwrap();

        assert_eq!(cache.block_range(), Some(42..=43));
        assert_eq!(cache.get(&key(1), txs_hash), Some(witness(1)));
        assert_eq!(cache.get(&key(2), txs_hash), Some(witness(2)));
        assert!(!dir.path().join("interrupted.tmp").exists());
    }

    #[test]
    fn insert_leaves_only_committed_files() {
        let dir = tempfile::tempdir().unwrap();
        let cache = WitnessCache::open(dir.path()).unwrap();

        cache.insert(1, key(1), B256::ZERO, &witness(1)).unwrap();

        let entry = WitnessCacheEntry { block_number: 1, transactions_hash: B256::ZERO, size: 0 };
        assert_eq!(files(dir.path()), vec![entry.file_name(&key(1))]);
        assert_eq!(
            WitnessCacheEntry::parse_file_name(&entry.file_name(&key(1))),
            Some((key(1), entry))
        );
    }

    #[tokio::test]
    async fn get_or_build_serves_cached_witness_without_building() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Arc::new(WitnessCache::open(dir.path()).unwrap());
        cache.insert(10, key(1), B256::ZERO, &witness(1)).unwrap();

        let mut built = false;
        let served = cache
            .get_or_build(10, key(1), B256::ZERO, async {
                built = true;
                Ok::<_, ()>(witness(2))
            })
            .await;

        assert_eq!(served, Ok(witness(1)));
        assert!(!built);
    }

    #[tokio::test]
    async fn get_or_build_caches_misses_inside_block_range_only() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Arc::new(WitnessCache::open(dir.path()).unwrap());
        cache.insert(10, key(1), B256::ZERO, &witness(1)).unwrap();
        cache.insert(20, key(2), B256::ZERO, &witness(2)).unwrap();

        let inside = cache.get_or_build(15, key(3), B256::ZERO, async { Ok::<_, ()>(witness(3)) });
        assert_eq!(inside.await, Ok(witness(3)));
        let outside = cache.get_or_build(21, key(4), B256::ZERO, async { Ok::<_, ()>(witness(4)) });
        assert_eq!(outside.await, Ok(witness(4)));
        let failed = cache.get_or_build(16, key(5), B256::ZERO, async { Err(()) });
        assert_eq!(failed.await, Err(()));

        assert_eq!(cache.get(&key(3), B256::ZERO), Some(witness(3)));
        assert!(!cache.contains(&key(4)));
        assert!(!cache.contains(&key(5)));
    }

    #[tokio::test]
    async fn get_or_build_rebuilds_on_transactions_hash_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Arc::new(WitnessCache::open(dir.path()).unwrap());
        cache.insert(10, key(1), B256::ZERO, &witness(1)).unwrap();

        let served = cache
            .get_or_build(10, key(1), B256::repeat_byte(1), async { Ok::<_, ()>(witness(2)) })
            .await;

        assert_eq!(served, Ok(witness(2)));
        assert_eq!(cache.get(&key(1), B256::repeat_byte(1)), Some(witness(2)));
    }

    #[test]
    fn corrupt_entry_is_dropped_as_miss() {
        let dir = tempfile::tempdir().unwrap();
        let cache = WitnessCache::open(dir.path()).unwrap();
        cache.insert(1, key(1), B256::ZERO, &witness(1)).unwrap();
        let entry = WitnessCacheEntry { block_number: 1, transactions_hash: B256::ZERO, size: 0 };
        fs::write(dir.path().join(entry.file_name(&key(1))), b"garbage").unwrap();

        assert_eq!(cache.get(&key(1), B256::ZERO), None);
        assert!(!cache.contains(&key(1)));
        assert!(files(dir.path()).is_empty());
    }
}
