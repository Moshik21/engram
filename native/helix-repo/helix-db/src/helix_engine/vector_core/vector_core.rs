use super::binary_heap::BinaryHeap;
use crate::{
    debug_println,
    helix_engine::{
        types::VectorError,
        vector_core::{
            hnsw::HNSW,
            utils::{Candidate, HeapOps, VectorFilter},
            vector::HVector,
            vector_without_data::VectorWithoutData,
        },
    },
    utils::{
        id::{splitmix64, stable_id_from_bytes, uuid_str, v6_uuid},
        properties::ImmutablePropertiesMap,
    },
};
use heed3::{
    Database, Env, RoTxn, RwTxn,
    byteorder::BE,
    types::{Bytes, U128, Unit},
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Pure, deterministic HNSW level from a vector id. Hashes the id to a uniform
/// f64 in (0,1] and applies the standard `(-ln(u) * m_l).floor()` transform, so
/// the level distribution matches the original random construction while being
/// reproducible across builds. Extracted from `get_new_level` so the
/// distribution is unit-testable without a `VectorCore`.
#[inline]
fn level_from_id(id: u128, m_l: f64) -> usize {
    let h = splitmix64((id as u64) ^ ((id >> 64) as u64));
    let u = ((h >> 11) as f64) / ((1u64 << 53) as f64);
    let r = if u <= 0.0 { f64::MIN_POSITIVE } else { u };
    (-r.ln() * m_l).floor() as usize
}

const DB_VECTORS: &str = "vectors"; // for vector data (v:)
const DB_VECTOR_DATA: &str = "vector_data"; // for vector data (v:)
const DB_HNSW_EDGES: &str = "hnsw_out_nodes"; // for hnsw out node data
const VECTOR_PREFIX: &[u8] = b"v:";
pub const ENTRY_POINT_KEY: &[u8] = b"entry_point";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWConfig {
    pub m: usize,             // max num of bi-directional links per element
    pub m_max_0: usize,       // max num of links for lower layers
    pub ef_construct: usize,  // size of the dynamic candidate list for construction
    pub m_l: f64,             // level generation factor
    pub ef: usize,            // search param, num of cands to search
    pub min_neighbors: usize, // for get_neighbors, always 512
}

impl HNSWConfig {
    /// Constructor for the configs of the HNSW vector similarity search algorithm
    /// - m (5 <= m <= 48): max num of bi-directional links per element
    /// - m_max_0 (2 * m): max num of links for level 0 (level that stores all vecs)
    /// - ef_construct (40 <= ef_construct <= 512): size of the dynamic candidate list
    ///   for construction
    /// - m_l (ln(1/m)): level generation factor (multiplied by a random number)
    /// - ef (10 <= ef <= 512): num of candidates to search
    pub fn new(m: Option<usize>, ef_construct: Option<usize>, ef: Option<usize>) -> Self {
        let m = m.unwrap_or(16).clamp(5, 48);
        let ef_construct = ef_construct.unwrap_or(128).clamp(40, 512);
        let ef = ef.unwrap_or(768).clamp(10, 512);

        Self {
            m,
            m_max_0: 2 * m,
            ef_construct,
            m_l: 1.0 / (m as f64).ln(),
            ef,
            min_neighbors: 512,
        }
    }
}

pub struct VectorCore {
    pub vectors_db: Database<Bytes, Bytes>,
    pub vector_properties_db: Database<U128<BE>, Bytes>,
    pub edges_db: Database<Bytes, Unit>,
    pub config: HNSWConfig,
}

impl VectorCore {
    pub fn new(env: &Env, txn: &mut RwTxn, config: HNSWConfig) -> Result<Self, VectorError> {
        let vectors_db = env.create_database(txn, Some(DB_VECTORS))?;
        let vector_properties_db = env
            .database_options()
            .types::<U128<BE>, Bytes>()
            .name(DB_VECTOR_DATA)
            .create(txn)?;
        let edges_db = env.create_database(txn, Some(DB_HNSW_EDGES))?;

        Ok(Self {
            vectors_db,
            vector_properties_db,
            edges_db,
            config,
        })
    }

    /// Vector key: [v, id, ]
    #[inline(always)]
    pub fn vector_key(id: u128, level: usize) -> Vec<u8> {
        [VECTOR_PREFIX, &id.to_be_bytes(), &level.to_be_bytes()].concat()
    }

    #[inline(always)]
    pub fn out_edges_key(source_id: u128, level: usize, sink_id: Option<u128>) -> Vec<u8> {
        match sink_id {
            Some(sink_id) => [
                source_id.to_be_bytes().as_slice(),
                level.to_be_bytes().as_slice(),
                sink_id.to_be_bytes().as_slice(),
            ]
            .concat()
            .to_vec(),
            None => [
                source_id.to_be_bytes().as_slice(),
                level.to_be_bytes().as_slice(),
            ]
            .concat()
            .to_vec(),
        }
    }

    #[inline]
    fn get_new_level(&self, id: u128) -> usize {
        // Deterministic per-id level: hash the (stable) id to a uniform f64 in
        // (0,1] and reuse the EXACT existing transform, so the exponential level
        // distribution — and thus hub/entry-point statistics and recall quality —
        // is preserved. Replaces the unseeded rand::rng() draw that made levels
        // differ on every rebuild. Top-53 bits give a uniform double; the u<=0
        // guard is strictly safer than rand (which could draw 0.0 -> ln(0)=-inf).
        level_from_id(id, self.config.m_l)
    }

    /// Derive a deterministic, build-independent vector id from the stable
    /// business key the caller already supplies in `properties` (EntityVec /
    /// GraphEmbedVec -> entity_id; EpisodeVec / CueVec / EpisodeChunk ->
    /// episode_id [+ chunk_index]). Label-prefixed so distinct vector types that
    /// share a key (e.g. EpisodeVec vs CueVec) never collide. Returns None when
    /// no recognized key is present (caller falls back to a fresh UUID).
    fn stable_vector_id(label: &str, properties: Option<&ImmutablePropertiesMap>) -> Option<u128> {
        let props = properties?;
        let key = ["entity_id", "episode_id", "id"]
            .iter()
            .find_map(|k| {
                props
                    .get(k)
                    .map(|v| v.inner_stringify())
                    .filter(|s| !s.is_empty())
            })?;
        let mut canonical = format!("{label}\x1f{key}");
        if let Some(ci) = props.get("chunk_index") {
            canonical.push('\x1f');
            canonical.push_str(&ci.inner_stringify());
        }
        Some(stable_id_from_bytes(canonical.as_bytes()))
    }

    #[inline]
    fn get_entry_point<'db: 'arena, 'arena: 'txn, 'txn>(
        &self,
        txn: &'txn RoTxn<'db>,
        label: &'arena str,
        arena: &'arena bumpalo::Bump,
    ) -> Result<HVector<'arena>, VectorError> {
        let ep_id = self.vectors_db.get(txn, ENTRY_POINT_KEY)?;
        if let Some(ep_id) = ep_id {
            let mut arr = [0u8; 16];
            let len = std::cmp::min(ep_id.len(), 16);
            arr[..len].copy_from_slice(&ep_id[..len]);
            self.get_raw_vector_data(txn, u128::from_be_bytes(arr), label, arena)
        } else {
            Err(VectorError::EntryPointNotFound)
        }
    }

    #[inline]
    fn set_entry_point(&self, txn: &mut RwTxn, entry: &HVector) -> Result<(), VectorError> {
        self.vectors_db
            .put(txn, ENTRY_POINT_KEY, &entry.id.to_be_bytes())
            .map_err(VectorError::from)?;
        Ok(())
    }

    #[inline(always)]
    pub fn put_vector<'arena>(
        &self,
        txn: &mut RwTxn,
        vector: &HVector<'arena>,
    ) -> Result<(), VectorError> {
        self.vectors_db
            .put(
                txn,
                &Self::vector_key(vector.id, vector.level),
                vector.vector_data_to_bytes()?,
            )
            .map_err(VectorError::from)?;
        self.vector_properties_db
            .put(txn, &vector.id, bincode::serialize(&vector)?.as_ref())?;
        Ok(())
    }

    #[inline(always)]
    fn get_neighbors<'db: 'arena, 'arena: 'txn, 'txn, F>(
        &self,
        txn: &'txn RoTxn<'db>,
        label: &'arena str,
        id: u128,
        level: usize,
        filter: Option<&[F]>,
        arena: &'arena bumpalo::Bump,
    ) -> Result<bumpalo::collections::Vec<'arena, HVector<'arena>>, VectorError>
    where
        F: Fn(&HVector<'arena>, &RoTxn<'db>) -> bool,
    {
        let out_key = Self::out_edges_key(id, level, None);
        let mut neighbors = bumpalo::collections::Vec::with_capacity_in(
            self.config.m_max_0.min(self.config.min_neighbors),
            arena,
        );

        let iter = self
            .edges_db
            .lazily_decode_data()
            .prefix_iter(txn, &out_key)?;

        let prefix_len = out_key.len();

        for result in iter {
            let (key, _) = result?;

            let mut arr = [0u8; 16];
            arr[..16].copy_from_slice(&key[prefix_len..(prefix_len + 16)]);
            let neighbor_id = u128::from_be_bytes(arr);

            if neighbor_id == id {
                continue;
            }
            let vector = self.get_raw_vector_data(txn, neighbor_id, label, arena)?;

            let passes_filters = match filter {
                Some(filter_slice) => filter_slice.iter().all(|f| f(&vector, txn)),
                None => true,
            };

            if passes_filters {
                neighbors.push(vector);
            }
        }
        neighbors.shrink_to_fit();

        Ok(neighbors)
    }

    #[inline(always)]
    fn set_neighbours<'db: 'arena, 'arena: 'txn, 'txn, 's>(
        &'db self,
        txn: &'txn mut RwTxn<'db>,
        id: u128,
        neighbors: &BinaryHeap<'arena, HVector<'arena>>,
        level: usize,
    ) -> Result<(), VectorError> {
        let prefix = Self::out_edges_key(id, level, None);

        let mut keys_to_delete: HashSet<Vec<u8>> = self
            .edges_db
            .prefix_iter(txn, prefix.as_ref())?
            .filter_map(|result| result.ok().map(|(key, _)| key.to_vec()))
            .collect();

        neighbors
            .iter()
            .try_for_each(|neighbor| -> Result<(), VectorError> {
                let neighbor_id = neighbor.id;
                if neighbor_id == id {
                    return Ok(());
                }

                let out_key = Self::out_edges_key(id, level, Some(neighbor_id));
                keys_to_delete.remove(&out_key);
                self.edges_db.put(txn, &out_key, &())?;

                let in_key = Self::out_edges_key(neighbor_id, level, Some(id));
                keys_to_delete.remove(&in_key);
                self.edges_db.put(txn, &in_key, &())?;

                Ok(())
            })?;

        for key in keys_to_delete {
            self.edges_db.delete(txn, &key)?;
        }

        Ok(())
    }

    fn select_neighbors<'db: 'arena, 'arena: 'txn, 'txn, 's, F>(
        &'db self,
        txn: &'txn RoTxn<'db>,
        label: &'arena str,
        query: &'s HVector<'arena>,
        mut cands: BinaryHeap<'arena, HVector<'arena>>,
        level: usize,
        should_extend: bool,
        filter: Option<&[F]>,
        arena: &'arena bumpalo::Bump,
    ) -> Result<BinaryHeap<'arena, HVector<'arena>>, VectorError>
    where
        F: Fn(&HVector<'arena>, &RoTxn<'db>) -> bool,
    {
        let m = self.config.m;

        if !should_extend {
            return Ok(cands.take_inord(m));
        }

        let mut visited: HashSet<u128> = HashSet::new();
        let mut result = BinaryHeap::with_capacity(arena, m * cands.len());
        for candidate in cands.iter() {
            for mut neighbor in
                self.get_neighbors(txn, label, candidate.id, level, filter, arena)?
            {
                if !visited.insert(neighbor.id) {
                    continue;
                }

                neighbor.set_distance(neighbor.distance_to(query)?);

                /*
                let passes_filters = match filter {
                    Some(filter_slice) => filter_slice.iter().all(|f| f(&neighbor, txn)),
                    None => true,
                };

                if passes_filters {
                    result.push(neighbor);
                }
                */

                if filter.is_none() || filter.unwrap().iter().all(|f| f(&neighbor, txn)) {
                    result.push(neighbor);
                }
            }
        }

        result.extend(cands);
        Ok(result.take_inord(m))
    }

    fn search_level<'db: 'arena, 'arena: 'txn, 'txn, 'q, F>(
        &self,
        txn: &'txn RoTxn<'db>,
        label: &'arena str,
        query: &'q HVector<'arena>,
        entry_point: &'q mut HVector<'arena>,
        ef: usize,
        level: usize,
        filter: Option<&[F]>,
        arena: &'arena bumpalo::Bump,
    ) -> Result<BinaryHeap<'arena, HVector<'arena>>, VectorError>
    where
        F: Fn(&HVector<'arena>, &RoTxn<'db>) -> bool,
    {
        let mut visited: HashSet<u128> = HashSet::new();
        let mut candidates: BinaryHeap<'arena, Candidate> =
            BinaryHeap::with_capacity(arena, self.config.ef_construct);
        let mut results: BinaryHeap<'arena, HVector<'arena>> = BinaryHeap::new(arena);

        entry_point.set_distance(entry_point.distance_to(query)?);
        candidates.push(Candidate {
            id: entry_point.id,
            distance: entry_point.get_distance(),
        });
        results.push(*entry_point);
        visited.insert(entry_point.id);

        while let Some(curr_cand) = candidates.pop() {
            if results.len() >= ef
                && results
                    .get_max()
                    .is_none_or(|f| curr_cand.distance > f.get_distance())
            {
                break;
            }

            let max_distance = if results.len() >= ef {
                results.get_max().map(|f| f.get_distance())
            } else {
                None
            };

            self.get_neighbors(txn, label, curr_cand.id, level, filter, arena)?
                .into_iter()
                .filter(|neighbor| visited.insert(neighbor.id))
                .filter_map(|mut neighbor| {
                    let distance = neighbor.distance_to(query).ok()?;

                    if max_distance.is_none_or(|max| distance < max) {
                        neighbor.set_distance(distance);
                        Some((neighbor, distance))
                    } else {
                        None
                    }
                })
                .for_each(|(neighbor, distance)| {
                    candidates.push(Candidate {
                        id: neighbor.id,
                        distance,
                    });

                    results.push(neighbor);

                    if results.len() > ef {
                        results = results.take_inord(ef);
                    }
                });
        }
        Ok(results)
    }

    pub fn num_inserted_vectors(&self, txn: &RoTxn) -> Result<u64, VectorError> {
        Ok(self.vectors_db.len(txn)?)
    }

    #[inline]
    pub fn get_vector_properties<'db: 'arena, 'arena: 'txn, 'txn>(
        &self,
        txn: &'txn RoTxn<'db>,
        id: u128,
        arena: &'arena bumpalo::Bump,
    ) -> Result<Option<VectorWithoutData<'arena>>, VectorError> {
        let vector: Option<VectorWithoutData<'arena>> =
            match self.vector_properties_db.get(txn, &id)? {
                Some(bytes) => Some(VectorWithoutData::from_bincode_bytes(arena, bytes, id)?),
                None => None,
            };

        if let Some(vector) = vector
            && vector.deleted
        {
            return Err(VectorError::VectorDeleted);
        }

        Ok(vector)
    }

    #[inline(always)]
    pub fn get_full_vector<'arena>(
        &self,
        txn: &RoTxn,
        id: u128,
        arena: &'arena bumpalo::Bump,
    ) -> Result<HVector<'arena>, VectorError> {
        let vector_data_bytes = self
            .vectors_db
            .get(txn, &Self::vector_key(id, 0))?
            .ok_or(VectorError::VectorNotFound(uuid_str(id, arena).to_string()))?;

        let properties_bytes = self.vector_properties_db.get(txn, &id)?;

        let vector = HVector::from_bincode_bytes(arena, properties_bytes, vector_data_bytes, id)?;
        if vector.deleted {
            return Err(VectorError::VectorDeleted);
        }
        Ok(vector)
    }

    #[inline(always)]
    pub fn get_raw_vector_data<'db: 'arena, 'arena: 'txn, 'txn>(
        &self,
        txn: &'txn RoTxn<'db>,
        id: u128,
        label: &'arena str,
        arena: &'arena bumpalo::Bump,
    ) -> Result<HVector<'arena>, VectorError> {
        let vector_data_bytes = self
            .vectors_db
            .get(txn, &Self::vector_key(id, 0))?
            .ok_or(VectorError::EntryPointNotFound)?;
        HVector::from_raw_vector_data(arena, vector_data_bytes, label, id)
    }

    /// Get all vectors from the database, optionally filtered by level
    pub fn get_all_vectors<'db: 'arena, 'arena: 'txn, 'txn>(
        &self,
        txn: &'txn RoTxn<'db>,
        level: Option<usize>,
        arena: &'arena bumpalo::Bump,
    ) -> Result<bumpalo::collections::Vec<'arena, HVector<'arena>>, VectorError> {
        let mut vectors = bumpalo::collections::Vec::new_in(arena);

        // Iterate over all vectors in the database
        let prefix_iter = self.vectors_db.prefix_iter(txn, VECTOR_PREFIX)?;

        for result in prefix_iter {
            let (key, _) = result?;

            // Extract id from the key: v: (2 bytes) + id (16 bytes) + level (8 bytes)
            if key.len() < VECTOR_PREFIX.len() + 16 {
                continue; // Skip malformed keys
            }

            let mut id_bytes = [0u8; 16];
            id_bytes.copy_from_slice(&key[VECTOR_PREFIX.len()..VECTOR_PREFIX.len() + 16]);
            let id = u128::from_be_bytes(id_bytes);

            // Get the full vector using the existing method
            match self.get_full_vector(txn, id, arena) {
                Ok(vector) => {
                    // Filter by level if specified
                    if let Some(lvl) = level {
                        if vector.level == lvl {
                            vectors.push(vector);
                        }
                    } else {
                        vectors.push(vector);
                    }
                }
                Err(_) => {
                    // Skip vectors that can't be loaded (e.g., deleted)
                    continue;
                }
            }
        }

        Ok(vectors)
    }
}

impl HNSW for VectorCore {
    fn search<'db, 'arena, 'txn, F>(
        &self,
        txn: &'txn RoTxn<'db>,
        query: &'arena [f64],
        k: usize,
        label: &'arena str,
        filter: Option<&'arena [F]>,
        should_trickle: bool,
        arena: &'arena bumpalo::Bump,
    ) -> Result<bumpalo::collections::Vec<'arena, HVector<'arena>>, VectorError>
    where
        F: Fn(&HVector<'arena>, &RoTxn<'db>) -> bool,
        'db: 'arena,
        'arena: 'txn,
    {
        let query = HVector::from_slice(label, 0, query);
        // let temp_arena = bumpalo::Bump::new();

        let mut entry_point = self.get_entry_point(txn, label, arena)?;

        let ef = self.config.ef;
        let curr_level = entry_point.level;
        // println!("curr_level: {curr_level}");
        for level in (1..=curr_level).rev() {
            let mut nearest = self.search_level(
                txn,
                label,
                &query,
                &mut entry_point,
                ef,
                level,
                match should_trickle {
                    true => filter,
                    false => None,
                },
                arena,
            )?;
            if let Some(closest) = nearest.pop() {
                entry_point = closest;
            }
        }
        // println!("entry_point: {entry_point:?}");
        let candidates = self.search_level(
            txn,
            label,
            &query,
            &mut entry_point,
            ef,
            0,
            match should_trickle {
                true => filter,
                false => None,
            },
            arena,
        )?;
        // println!("candidates");
        let results = candidates.to_vec_with_filter::<F, true>(
            k,
            filter,
            label,
            txn,
            self.vector_properties_db,
            arena,
        )?;

        debug_println!("vector search found {} results", results.len());
        Ok(results)
    }

    fn insert<'db, 'arena, 'txn, F>(
        &'db self,
        txn: &'txn mut RwTxn<'db>,
        label: &'arena str,
        data: &'arena [f64],
        properties: Option<ImmutablePropertiesMap<'arena>>,
        arena: &'arena bumpalo::Bump,
    ) -> Result<HVector<'arena>, VectorError>
    where
        F: Fn(&HVector<'arena>, &RoTxn<'db>) -> bool,
        'db: 'arena,
        'arena: 'txn,
    {
        // Deterministic id from the stable business key (falls back to a fresh
        // UUID only when no recognized key is present) so the persisted graph —
        // ids are the LMDB edge keys, so they also fix neighbor-iteration order —
        // and the per-id level are build-independent.
        let id = Self::stable_vector_id(label, properties.as_ref()).unwrap_or_else(v6_uuid);
        let mut query = HVector::from_slice_with_id(label, 0, data, id);
        query.properties = properties;
        let new_level = self.get_new_level(query.id);
        self.put_vector(txn, &query)?;

        query.level = new_level;

        let entry_point = match self.get_entry_point(txn, label, arena) {
            Ok(ep) => ep,
            Err(_) => {
                // TODO: use proper error handling
                self.set_entry_point(txn, &query)?;
                query.set_distance(0.0);

                return Ok(query);
            }
        };

        let l = entry_point.level;
        let ep_id = entry_point.id;
        let mut curr_ep = entry_point;
        for level in (new_level + 1..=l).rev() {
            let mut nearest =
                self.search_level::<F>(txn, label, &query, &mut curr_ep, 1, level, None, arena)?;
            curr_ep = nearest.pop().ok_or(VectorError::VectorCoreError(
                "emtpy search result".to_string(),
            ))?;
        }

        for level in (0..=l.min(new_level)).rev() {
            let nearest = self.search_level::<F>(
                txn,
                label,
                &query,
                &mut curr_ep,
                self.config.ef_construct,
                level,
                None,
                arena,
            )?;
            curr_ep = *nearest.peek().ok_or(VectorError::VectorCoreError(
                "emtpy search result".to_string(),
            ))?;

            let neighbors =
                self.select_neighbors::<F>(txn, label, &query, nearest, level, true, None, arena)?;
            self.set_neighbours(txn, query.id, &neighbors, level)?;

            for e in neighbors {
                let id = e.id;
                let e_conns = BinaryHeap::from(
                    arena,
                    self.get_neighbors::<F>(txn, label, id, level, None, arena)?,
                );
                let e_new_conn = self
                    .select_neighbors::<F>(txn, label, &query, e_conns, level, true, None, arena)?;
                self.set_neighbours(txn, id, &e_new_conn, level)?;
            }
        }

        // Update the entry point on a strictly higher level, OR break a max-level
        // tie by smaller id. The id tie-break makes the entry point converge to
        // the smallest-id node at the max level regardless of insertion order —
        // removing an insertion-order dependence in the otherwise-deterministic
        // graph (transitive: any smaller-id max-level node inserted later wins).
        if new_level > l || (new_level == l && query.id < ep_id) {
            self.set_entry_point(txn, &query)?;
        }

        debug_println!("vector inserted with id {}", query.id);
        Ok(query)
    }

    fn delete(&self, txn: &mut RwTxn, id: u128, arena: &bumpalo::Bump) -> Result<(), VectorError> {
        match self.get_vector_properties(txn, id, arena)? {
            Some(mut properties) => {
                debug_println!("properties: {properties:?}");
                if properties.deleted {
                    return Err(VectorError::VectorAlreadyDeleted(id.to_string()));
                }

                properties.deleted = true;
                self.vector_properties_db.put(
                    txn,
                    &id,
                    bincode::serialize(&properties)?.as_ref(),
                )?;
                debug_println!("vector deleted with id {}", &id);
                Ok(())
            }
            None => Err(VectorError::VectorNotFound(id.to_string())),
        }
    }
}

#[cfg(test)]
mod determinism_tests {
    use super::level_from_id;
    use crate::utils::id::{fnv1a64, splitmix64, stable_id_from_bytes};

    #[test]
    fn stable_id_is_deterministic_and_distinct() {
        // Same bytes -> same id on every call (build-independent).
        assert_eq!(
            stable_id_from_bytes(b"EpisodeVec\x1fep_123"),
            stable_id_from_bytes(b"EpisodeVec\x1fep_123")
        );
        // Distinct keys -> distinct ids; label prefix keeps a shared business key
        // (episode_id) distinct across vector types.
        assert_ne!(
            stable_id_from_bytes(b"EpisodeVec\x1fep_123"),
            stable_id_from_bytes(b"CueVec\x1fep_123")
        );
        assert_ne!(
            stable_id_from_bytes(b"EpisodeVec\x1fep_123"),
            stable_id_from_bytes(b"EpisodeVec\x1fep_124")
        );
        // splitmix64 is a pure finalizer.
        assert_eq!(splitmix64(42), splitmix64(42));
        assert_ne!(fnv1a64(0, b"a"), fnv1a64(0, b"b"));
    }

    #[test]
    fn level_is_deterministic_per_id() {
        // The dominant fix: a given id always maps to the same level (was a fresh
        // rand draw every build).
        let m_l = 1.0 / (2.0_f64).ln();
        for id in [0u128, 1, 7, u128::MAX, 0x1234_5678_9ABC_DEF0] {
            assert_eq!(level_from_id(id, m_l), level_from_id(id, m_l));
        }
    }

    #[test]
    fn level_distribution_is_preserved() {
        // Recall-safety guard: hashing the id instead of drawing rand must keep
        // the exponential profile. Level 0 should dominate (~1-1/e of nodes) and
        // the mean should sit near m_l, matching the original construction.
        let m_l = 1.0 / (2.0_f64).ln(); // ~1.4427, the canonical HNSW factor
        let n = 50_000u128;
        let mut level0 = 0u64;
        let mut sum = 0u64;
        let mut max_level = 0usize;
        for i in 0..n {
            // Spread ids the way stable_id_from_bytes would (not sequential).
            let id = stable_id_from_bytes(format!("EpisodeVec\x1fep_{i}").as_bytes());
            let lvl = level_from_id(id, m_l);
            if lvl == 0 {
                level0 += 1;
            }
            sum += lvl as u64;
            max_level = max_level.max(lvl);
        }
        let frac_level0 = level0 as f64 / n as f64;
        let mean = sum as f64 / n as f64;
        // P(level==0) = 1 - e^{-1/m_l} ~= 0.5 for m_l=1/ln2; allow generous slack.
        assert!(
            (0.40..0.60).contains(&frac_level0),
            "level-0 fraction {frac_level0} outside expected ~0.5 band"
        );
        // Mean level of a geometric(p=0.5) tail ~= 1.0; guard a sane band.
        assert!(
            (0.6..1.6).contains(&mean),
            "mean level {mean} outside expected band"
        );
        // Sanity: the tower has some height but is not absurd for 50k nodes.
        assert!((5..40).contains(&max_level), "max level {max_level} implausible");
    }
}
