use std::collections::HashSet;
use std::sync::atomic::{self, AtomicUsize};
use std::time::Instant;
use std::{collections::HashMap, sync::Arc};

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use core_affinity::CoreId;
use reqwest::header::CONTENT_TYPE;
use sonic_rs::{JsonContainerTrait, JsonValueTrait};
use tracing::{info, trace, warn};

use crate::protocol::Format;
use crate::protocol::request::RequestType;

use super::router::router::{HandlerFn, HelixRouter};
#[cfg(feature = "dev-instance")]
use crate::helix_gateway::builtin::all_nodes_and_edges::nodes_edges_handler;
#[cfg(feature = "dev-instance")]
use crate::helix_gateway::builtin::node_by_id::node_details_handler;
#[cfg(feature = "dev-instance")]
use crate::helix_gateway::builtin::node_connections::node_connections_handler;
#[cfg(feature = "dev-instance")]
use crate::helix_gateway::builtin::nodes_by_label::nodes_by_label_handler;
use crate::helix_gateway::introspect_schema::introspect_schema_handler;
use crate::helix_gateway::worker_pool::WorkerPool;
use crate::protocol;
use crate::{
    helix_engine::traversal_core::{HelixGraphEngine, HelixGraphEngineOpts},
    helix_gateway::mcp::mcp::MCPHandlerFn,
};

pub struct GatewayOpts {}

impl GatewayOpts {
    pub const DEFAULT_WORKERS_PER_CORE: usize = 8;
}

pub struct HelixGateway {
    pub(crate) address: String,
    pub(crate) workers_per_core: usize,
    pub(crate) graph_access: Arc<HelixGraphEngine>,
    pub(crate) router: Arc<HelixRouter>,
    pub(crate) opts: Option<HelixGraphEngineOpts>,
    pub(crate) cluster_id: Option<String>,
}

impl HelixGateway {
    pub fn new(
        address: &str,
        graph_access: Arc<HelixGraphEngine>,
        workers_per_core: usize,
        routes: Option<HashMap<String, HandlerFn>>,
        mcp_routes: Option<HashMap<String, MCPHandlerFn>>,
        write_routes: Option<HashSet<String>>,
        opts: Option<HelixGraphEngineOpts>,
    ) -> HelixGateway {
        let router = Arc::new(HelixRouter::new(routes, mcp_routes, write_routes));
        let cluster_id = std::env::var("HELIX_CLUSTER_ID").ok();
        HelixGateway {
            address: address.to_string(),
            graph_access,
            router,
            workers_per_core,
            opts,
            cluster_id,
        }
    }

    pub fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        trace!("Starting Helix Gateway");

        let all_core_ids = core_affinity::get_core_ids().expect("unable to get core IDs");

        let all_core_ids = match std::env::var("HELIX_CORES_OVERRIDE") {
            Ok(val) => {
                let override_count: usize = val
                    .parse()
                    .expect("HELIX_CORES_OVERRIDE must be a valid number");
                if override_count > all_core_ids.len() {
                    warn!(
                        "HELIX_CORES_OVERRIDE ({}) exceeds available cores ({}), using all cores",
                        override_count,
                        all_core_ids.len()
                    );
                    all_core_ids
                } else {
                    all_core_ids.into_iter().take(override_count).collect()
                }
            }
            Err(_) => all_core_ids,
        };

        info!(
            "Worker pool initialized: {} cores, {} worker threads, 1 writer thread",
            all_core_ids.len(),
            all_core_ids.len() * self.workers_per_core
        );

        let tokio_core_ids = all_core_ids.clone();
        let tokio_core_setter = Arc::new(CoreSetter::new(tokio_core_ids, 1));

        let rt = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(tokio_core_setter.num_threads())
                .on_thread_unpark(move || Arc::clone(&tokio_core_setter).set_current_once())
                .enable_all()
                .build()?,
        );

        let worker_core_ids = all_core_ids.clone();
        let worker_core_setter = Arc::new(CoreSetter::new(worker_core_ids, self.workers_per_core));

        let worker_pool = WorkerPool::new(
            worker_core_setter,
            Arc::clone(&self.graph_access),
            Arc::clone(&self.router),
            Arc::clone(&rt),
        );

        let mut axum_app = axum::Router::new();

        axum_app = axum_app
            .route("/batch", post(batch_handler))
            .route("/{*path}", post(post_handler))
            .route("/introspect", get(introspect_schema_handler));

        #[cfg(feature = "dev-instance")]
        {
            axum_app = axum_app
                .route("/nodes-edges", get(nodes_edges_handler))
                .route("/nodes-by-label", get(nodes_by_label_handler))
                .route("/node-connections", get(node_connections_handler))
                .route("/node-details", get(node_details_handler));
        }

        let app_state = Arc::new(AppState {
            worker_pool,
            schema_json: self
                .opts
                .and_then(|o| o.config.schema.map(Bytes::from)),
            cluster_id: self.cluster_id,
        });
        let axum_app = axum_app.with_state(Arc::clone(&app_state));
        let http_address = self.address.clone();

        rt.block_on(async move {
            // Initialize metrics system
            helix_metrics::init_metrics_system();

            #[cfg(feature = "grpc")]
            {
                use crate::helix_gateway::grpc::proto::helix_db_server::HelixDbServer;
                use crate::helix_gateway::grpc::service::HelixGrpcService;
                let grpc_addr = derive_grpc_address(&http_address);
                let grpc_state = Arc::clone(&app_state);
                tokio::spawn(async move {
                    let svc = HelixGrpcService::new(grpc_state);
                    info!("gRPC server listening on {}", grpc_addr);
                    if let Err(e) = tonic::transport::Server::builder()
                        .add_service(HelixDbServer::new(svc))
                        .serve(grpc_addr.parse().expect("valid gRPC address"))
                        .await
                    {
                        warn!("gRPC server exited: {}", e);
                    }
                });
            }

            let listener = tokio::net::TcpListener::bind(self.address)
                .await
                .expect("Failed to bind listener");
            info!("Listener has been bound, starting server");
            axum::serve(listener, axum_app)
                .with_graceful_shutdown(shutdown_signal())
                .await
                .expect("Failed to serve");

            // Shutdown metrics system to flush all pending events
            info!("Shutting down metrics system...");
            let shutdown_result = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                helix_metrics::shutdown_metrics_system(),
            )
            .await;

            match shutdown_result {
                Ok(_) => info!("Metrics system shutdown complete"),
                Err(_) => warn!("Metrics system shutdown timed out after 5 seconds"),
            }
        });

        Ok(())
    }
}

async fn shutdown_signal() {
    // Respond to either Ctrl-C (SIGINT) or SIGTERM (e.g. `kill` or systemd stop)
    #[cfg(unix)]
    {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Received Ctrl-C, starting graceful shutdown…");
            }
            _ = sigterm() => {
                info!("Received SIGTERM, starting graceful shutdown…");
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        info!("Received Ctrl-C, starting graceful shutdown…");
    }
}

#[cfg(unix)]
async fn sigterm() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    term.recv().await;
}

async fn post_handler(
    State(state): State<Arc<AppState>>,
    req: protocol::request::Request,
) -> axum::http::Response<Body> {
    let start_time = Instant::now();
    #[cfg(feature = "api-key")]
    {
        use crate::helix_gateway::key_verification::verify_key;
        if let Err(e) = verify_key(req.api_key.as_ref().unwrap()) {
            info!(?e, "Invalid API key");
            helix_metrics::log_event(
                helix_metrics::events::EventType::InvalidApiKey,
                helix_metrics::events::InvalidApiKeyEvent {
                    cluster_id: state.cluster_id.clone(),
                    time_taken_usec: start_time.elapsed().as_micros() as u32,
                },
            );
            return e.into_response();
        }
    }
    let input_body = if *helix_metrics::METRICS_ENABLED {
        Some(req.body.clone())
    } else {
        None
    };
    let query_name = req.name.clone();
    let res = state.worker_pool.process(req).await;

    match res {
        Ok(r) => {
            #[cfg(any(feature = "dev-instance", feature = "production"))]
            {
                let resp_str = String::from_utf8_lossy(&r.body);
                info!(query = %query_name, response = %resp_str, "Response");
            }
            if !*helix_metrics::METRICS_ENABLED {
                return r.into_response();
            }
            helix_metrics::log_event(
                helix_metrics::events::EventType::QuerySuccess,
                helix_metrics::events::QuerySuccessEvent {
                    cluster_id: state.cluster_id.clone(),
                    query_name,
                    time_taken_usec: start_time.elapsed().as_micros() as u32,
                },
            );
            r.into_response()
        }
        Err(e) => {
            info!(query = %query_name, error = ?e, "Error response");
            if !*helix_metrics::METRICS_ENABLED {
                return e.into_response();
            }
            helix_metrics::log_event(
                helix_metrics::events::EventType::QueryError,
                helix_metrics::events::QueryErrorEvent {
                    cluster_id: state.cluster_id.clone(),
                    query_name,
                    input_json: input_body
                        .as_ref()
                        .and_then(|body| std::str::from_utf8(body.as_ref()).ok())
                        .map(str::to_owned),
                    output_json: sonic_rs::to_string(&e).ok(),
                    time_taken_usec: start_time.elapsed().as_micros() as u32,
                },
            );
            e.into_response()
        }
    }
}

async fn batch_handler(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> axum::http::Response<Body> {
    let start_time = Instant::now();
    let batch_req: sonic_rs::Value = match sonic_rs::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            let err = br#"{"error":"Invalid JSON in batch request","code":"BAD_REQUEST"}"#;
            return axum::response::Response::builder()
                .status(400)
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(err.to_vec()))
                .expect("static response should always build");
        }
    };
    let queries = match batch_req.get("queries").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => {
            let err = br#"{"error":"Missing or invalid 'queries' array","code":"BAD_REQUEST"}"#;
            return axum::response::Response::builder()
                .status(400)
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(err.to_vec()))
                .expect("static response should always build");
        }
    };
    if queries.len() > 50 {
        let err = br#"{"error":"Batch size exceeds maximum of 50 queries","code":"BAD_REQUEST"}"#;
        return axum::response::Response::builder()
            .status(400)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(err.to_vec()))
            .expect("static response should always build");
    }
    let mut futures = Vec::with_capacity(queries.len());
    for query in queries {
        let name = match query.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => {
                futures.push(tokio::spawn(async {
                    Err::<protocol::response::Response, String>(
                        "Missing 'name' field in query".to_string(),
                    )
                }));
                continue;
            }
        };
        let body_bytes = match query.get("body") {
            Some(b) => Bytes::from(sonic_rs::to_vec(b).unwrap_or_default()),
            None => Bytes::new(),
        };
        let req = protocol::Request {
            name,
            req_type: RequestType::Query,
            api_key: None,
            body: body_bytes,
            in_fmt: Format::Json,
            out_fmt: Format::Json,
        };
        let state_ref = Arc::clone(&state);
        futures.push(tokio::spawn(async move {
            match state_ref.worker_pool.process(req).await {
                Ok(resp) => Ok(resp),
                Err(e) => Err(e.to_string()),
            }
        }));
    }
    let mut results = Vec::with_capacity(futures.len());
    for fut in futures {
        let join_result = fut.await;
        match join_result {
            Ok(Ok(resp)) => {
                let body_val: sonic_rs::Value =
                    sonic_rs::from_slice(&resp.body).unwrap_or_default();
                results.push(BatchResult { status: 200, body: Some(body_val), error: None });
            }
            Ok(Err(err_str)) => {
                results.push(BatchResult { status: 500, body: None, error: Some(err_str) });
            }
            Err(join_err) => {
                results.push(BatchResult { status: 500, body: None, error: Some(join_err.to_string()) });
            }
        }
    }
    let response_body = BatchResponse { results };
    let response_bytes = sonic_rs::to_vec(&response_body).unwrap_or_default();
    if *helix_metrics::METRICS_ENABLED {
        helix_metrics::log_event(
            helix_metrics::events::EventType::QuerySuccess,
            helix_metrics::events::QuerySuccessEvent {
                cluster_id: state.cluster_id.clone(),
                query_name: "batch".to_string(),
                time_taken_usec: start_time.elapsed().as_micros() as u32,
            },
        );
    }
    axum::response::Response::builder()
        .status(200)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(response_bytes))
        .expect("Should be able to construct batch response")
}

pub struct AppState {
    pub worker_pool: WorkerPool,
    pub schema_json: Option<Bytes>,
    pub cluster_id: Option<String>,
}

#[derive(serde::Serialize)]
struct BatchResponse {
    results: Vec<BatchResult>,
}

#[derive(serde::Serialize)]
struct BatchResult {
    status: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    body: Option<sonic_rs::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[cfg(feature = "grpc")]
fn derive_grpc_address(http_addr: &str) -> String {
    if let Some((host, port_str)) = http_addr.rsplit_once(':') {
        if let Ok(port) = port_str.parse::<u16>() {
            return format!("{}:{}", host, port + 1);
        }
    }
    format!("{}:6970", http_addr)
}

pub struct CoreSetter {
    pub(crate) cores: Vec<CoreId>,
    pub(crate) threads_per_core: usize,
    pub(crate) incrementing_index: AtomicUsize,
}

impl CoreSetter {
    pub fn new(cores: Vec<CoreId>, threads_per_core: usize) -> Self {
        Self {
            cores,
            threads_per_core,
            incrementing_index: AtomicUsize::new(0),
        }
    }

    pub fn num_threads(&self) -> usize {
        self.cores.len() * self.threads_per_core
    }

    pub fn set_current(self: Arc<Self>) {
        let curr_idx = self
            .incrementing_index
            .fetch_add(1, atomic::Ordering::SeqCst);

        let core_index = curr_idx / self.threads_per_core;
        match self.cores.get(core_index) {
            Some(c) => {
                core_affinity::set_for_current(*c);
            }
            None => warn!(
                "CoreSetter::set_current called more times than cores.len() * threads_per_core. Core affinity not set"
            ),
        };
    }

    pub fn set_current_once(self: Arc<Self>) {
        use std::sync::OnceLock;

        thread_local! {
            static CORE_SET: OnceLock<()> = const { OnceLock::new() };
        }

        CORE_SET.with(|flag| {
            flag.get_or_init(move || self.set_current());
        });
    }
}
