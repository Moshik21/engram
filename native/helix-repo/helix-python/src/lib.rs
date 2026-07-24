// Include the generated query handlers at crate root level FIRST.
// queries.rs uses top-level `use` statements and `inventory::submit!` which
// must be at crate scope for the handler registration to work.
include!(concat!(env!("OUT_DIR"), "/queries.rs"));

// PyO3 imports (after queries.rs to avoid duplicate import conflicts)
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use axum::body::Bytes;

use helix_db::helix_engine::traversal_core::{HelixGraphEngine, HelixGraphEngineOpts};
use helix_db::helix_engine::storage_core::version_info::VersionInfo;
use helix_db::helix_gateway::gateway::CoreSetter;
use helix_db::helix_gateway::router::router::HelixRouter;
use helix_db::helix_gateway::worker_pool::WorkerPool;
use helix_db::protocol::Request;
use helix_db::protocol::request::RequestType;
// HashMap is already imported via the generated queries.rs include

#[pyclass]
struct HelixEngine {
    graph: Arc<HelixGraphEngine>,
    worker_pool: WorkerPool,
    router: Arc<HelixRouter>,
    rt: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl HelixEngine {
    #[new]
    #[pyo3(signature = (data_dir=None, num_workers=None, bm25_field_filters=None))]
    fn new(
        data_dir: Option<String>,
        num_workers: Option<usize>,
        bm25_field_filters: Option<HashMap<String, Vec<String>>>,
    ) -> PyResult<Self> {
        // Resolve data directory
        let path = data_dir
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
                    .join(".helix")
                    .join("engram-native")
            });

        // Ensure data directory exists
        std::fs::create_dir_all(&path)
            .map_err(|e| PyRuntimeError::new_err(format!("Cannot create data dir: {e}")))?;

        // Get config from generated queries and apply BM25 field filters
        let mut config = config().unwrap_or_default();
        if bm25_field_filters.is_some() {
            config.bm25_field_filters = bm25_field_filters;
        }

        // Create engine
        let opts = HelixGraphEngineOpts {
            path: path.to_str().unwrap().to_string(),
            config,
            version_info: VersionInfo(HashMap::new()),
        };

        let graph = Arc::new(
            HelixGraphEngine::new(opts)
                .map_err(|e| PyRuntimeError::new_err(format!("Engine init failed: {e}")))?
        );

        // Build router from build-script-generated route map (bypasses inventory for cdylib compat)
        let (query_routes, write_routes) = register_routes();
        let route_count = query_routes.len();
        let router = Arc::new(HelixRouter::new(
            Some(query_routes),
            None, // No MCP routes for in-process
            Some(write_routes),
        ));

        // Build tokio runtime for async operations (continuation futures)
        let worker_count = num_workers.unwrap_or(4).max(2);
        let rt = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2) // Minimal - just for continuations
                .enable_all()
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("Runtime init: {e}")))?
        );

        // Build worker pool
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        let mut effective_cores: Vec<core_affinity::CoreId> = if core_ids.is_empty() {
            vec![core_affinity::CoreId { id: 0 }; worker_count]
        } else {
            let take = worker_count.min(core_ids.len());
            core_ids.into_iter().take(take).collect()
        };

        // Ensure even number of workers for WorkerPool parity requirement
        let total_workers = effective_cores.len();
        if total_workers < 2 {
            effective_cores = vec![core_affinity::CoreId { id: 0 }; 2];
        } else if total_workers % 2 != 0 {
            // Add one more core to make it even
            let last = effective_cores.last().copied().unwrap_or(core_affinity::CoreId { id: 0 });
            effective_cores.push(last);
        }

        let workers_per_core = 1;
        let core_setter = Arc::new(CoreSetter::new(effective_cores, workers_per_core));

        let worker_pool = WorkerPool::new(
            core_setter,
            Arc::clone(&graph),
            Arc::clone(&router),
            Arc::clone(&rt),
        );

        eprintln!(
            "helix_native: engine initialized (data_dir={}, routes={})",
            path.display(),
            route_count,
        );

        Ok(HelixEngine {
            graph,
            worker_pool,
            router,
            rt,
        })
    }

    /// Execute a single query. Blocks and releases the GIL.
    fn query(&self, py: Python<'_>, name: String, body_json: String) -> PyResult<String> {
        let request = Request {
            name: name.clone(),
            req_type: RequestType::Query,
            api_key: None,
            body: Bytes::from(body_json.into_bytes()),
            in_fmt: Format::Json,
            out_fmt: Format::Json,
        };

        let rt = Arc::clone(&self.rt);

        // Release GIL during Rust execution
        py.allow_threads(|| {
            rt.block_on(async {
                match self.worker_pool.process(request).await {
                    Ok(resp) => String::from_utf8(resp.body)
                        .map_err(|e| PyRuntimeError::new_err(format!("UTF-8 error: {e}"))),
                    Err(e) => Err(PyRuntimeError::new_err(format!("Query '{name}' failed: {e}"))),
                }
            })
        })
    }

    /// Execute multiple queries concurrently. Returns list of JSON strings.
    fn batch(&self, py: Python<'_>, queries: Vec<(String, String)>) -> PyResult<Vec<String>> {
        let requests: Vec<Request> = queries
            .into_iter()
            .map(|(name, body)| Request {
                name,
                req_type: RequestType::Query,
                api_key: None,
                body: Bytes::from(body.into_bytes()),
                in_fmt: Format::Json,
                out_fmt: Format::Json,
            })
            .collect();

        let rt = Arc::clone(&self.rt);

        py.allow_threads(|| {
            rt.block_on(async {
                let mut handles = Vec::with_capacity(requests.len());
                for req in requests {
                    handles.push(self.worker_pool.process(req));
                }

                let mut results = Vec::with_capacity(handles.len());
                for handle in handles {
                    match handle.await {
                        Ok(resp) => results.push(
                            String::from_utf8(resp.body).unwrap_or_default()
                        ),
                        Err(e) => results.push(
                            format!("{{\"error\":\"{}\"}}", e.to_string().replace('"', "\\\""))
                        ),
                    }
                }
                Ok(results)
            })
        })
    }

    /// Check if a route name is registered.
    fn has_route(&self, name: &str) -> bool {
        self.router.routes.contains_key(name)
    }

    /// List all registered route names.
    fn list_routes(&self) -> Vec<String> {
        self.router.routes.keys().cloned().collect()
    }

    /// Write a compacting copy of the LMDB environment to `dest_dir/data.mdb`.
    ///
    /// LMDB never returns freed pages to the OS, so a brain that has seen heavy
    /// write churn keeps paying (RAM residency, page faults) for pages it no
    /// longer stores. The copy omits free pages and renumbers the rest; graph,
    /// HNSW vectors and BM25 all live in this one env, so a single copy
    /// reclaims all three. Returns the byte size of the written file.
    ///
    /// The caller must guarantee there are no concurrent writers (shell down +
    /// brain flock held) — a copy taken under writes is not crash-consistent.
    fn compact(&self, py: Python<'_>, dest_dir: String) -> PyResult<u64> {
        let dest = std::path::PathBuf::from(&dest_dir).join("data.mdb");
        py.allow_threads(|| {
            let file = self
                .graph
                .storage
                .graph_env
                .copy_to_path(&dest, heed3::CompactionOption::Enabled)
                .map_err(|e| PyRuntimeError::new_err(format!("Compaction failed: {e}")))?;
            file.metadata()
                .map(|meta| meta.len())
                .map_err(|e| PyRuntimeError::new_err(format!("Cannot stat {dest:?}: {e}")))
        })
    }

    /// Graceful shutdown.
    fn close(&self) {
        // WorkerPool drop handles cleanup via flume channel disconnect
    }
}

// Auto-generated route map from build.rs - bypasses inventory for cdylib compatibility
include!(concat!(env!("OUT_DIR"), "/route_map.rs"));

/// Python module definition.
#[pymodule]
fn helix_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HelixEngine>()?;
    Ok(())
}
