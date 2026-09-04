use crate::{
    helix_engine::{
        bm25::bm25::BM25,
        traversal_core::{
            LMDB_STRING_HEADER_LENGTH, traversal_iter::RoTraversalIterator,
            traversal_value::TraversalValue,
        },
        types::GraphError,
    },
    utils::items::Node,
};

pub trait SearchBM25Adapter<'db, 'arena, 'txn>:
    Iterator<Item = Result<TraversalValue<'arena>, GraphError>>
{
    fn search_bm25<K>(
        self,
        label: &'arena str,
        query: &str,
        k: K,
    ) -> Result<
        RoTraversalIterator<
            'db,
            'arena,
            'txn,
            impl Iterator<Item = Result<TraversalValue<'arena>, GraphError>>,
        >,
        GraphError,
    >
    where
        K: TryInto<usize>,
        K::Error: std::fmt::Debug;
}

impl<'db, 'arena, 'txn, I: Iterator<Item = Result<TraversalValue<'arena>, GraphError>>>
    SearchBM25Adapter<'db, 'arena, 'txn> for RoTraversalIterator<'db, 'arena, 'txn, I>
{
    fn search_bm25<K>(
        self,
        label: &'arena str,
        query: &str,
        k: K,
    ) -> Result<
        RoTraversalIterator<
            'db,
            'arena,
            'txn,
            impl Iterator<Item = Result<TraversalValue<'arena>, GraphError>>,
        >,
        GraphError,
    >
    where
        K: TryInto<usize>,
        K::Error: std::fmt::Debug,
    {
        // The BM25 index spans every label. Cutting to `k` BEFORE the label
        // filter returned a page that was mostly other labels: measured
        // 2026-09-03 on an 8.6k-episode brain, `SearchBM25<Episode>("Engram", 50)`
        // yielded 7 episodes and `("semanticize", 50)` yielded 0 while 30
        // episodes contain the word -- the top 50 across all labels were cues
        // and chunks. The index already scores every matching document before
        // it truncates, so fetching a wider slice costs only by-id label
        // checks; the page is then cut to `k` AFTER the filter.
        let k_usize: usize = k.try_into().unwrap();
        let fetch = k_usize.saturating_mul(16).max(k_usize);
        let results = match self.storage.bm25.as_ref() {
            Some(s) => s.search(self.txn, query, fetch, self.arena)?,
            None => return Err(GraphError::from("BM25 not enabled!")),
        };

        let label_as_bytes = label.as_bytes();
        let iter = results.into_iter().filter_map(move |(id, score)| {
            if let Ok(Some(value)) = self.storage.nodes_db.get(self.txn, &id) {
            assert!(
                value.len() >= LMDB_STRING_HEADER_LENGTH,
                "value length does not contain header which means the `label` field was missing from the node on insertion"
            );
            let length_of_label_in_lmdb =
                u64::from_le_bytes(value[..LMDB_STRING_HEADER_LENGTH].try_into().unwrap()) as usize;

            if length_of_label_in_lmdb != label.len() {
                return None;
            }

            assert!(
                value.len() >= length_of_label_in_lmdb + LMDB_STRING_HEADER_LENGTH,
                "value length is not at least the header length plus the label length meaning there has been a corruption on node insertion"
            );
            let label_in_lmdb = &value[LMDB_STRING_HEADER_LENGTH
                ..LMDB_STRING_HEADER_LENGTH + length_of_label_in_lmdb];

            if label_in_lmdb == label_as_bytes {
                match Node::<'arena>::from_bincode_bytes(id, value, self.arena) {
                    Ok(node) => {
                        return Some(Ok(TraversalValue::NodeWithScore { node, score: score as f64 }));
                    }
                    Err(e) => {
                        println!("{} Error decoding node: {:?}", line!(), e);
                        return Some(Err(GraphError::ConversionError(e.to_string())));
                    }
                }
            } else {
                return None;
            }
            }
            None
        }).take(k_usize);

        Ok(RoTraversalIterator {
            storage: self.storage,
            arena: self.arena,
            txn: self.txn,
            inner: iter,
        })
    }
}
