use std::sync::Arc;
use axum::body::Bytes;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use tracing::warn;
use crate::helix_gateway::gateway::AppState;
use crate::protocol;
use crate::protocol::request::RequestType;
use crate::protocol::Format;
use super::proto::{
    helix_db_server::HelixDb, BatchRequest, BatchResponse, QueryRequest, QueryResponse,
};

pub struct HelixGrpcService {
    state: Arc<AppState>,
}

impl HelixGrpcService {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
    fn to_protocol_request(req: &QueryRequest) -> protocol::Request {
        protocol::Request {
            name: req.name.clone(),
            req_type: match req.req_type { 1 => RequestType::MCP, _ => RequestType::Query },
            api_key: if req.api_key.is_empty() { None } else { Some(req.api_key.clone()) },
            body: Bytes::from(req.body.clone()),
            in_fmt: Format::Json,
            out_fmt: Format::Json,
        }
    }
    async fn execute_one(&self, req: &QueryRequest) -> QueryResponse {
        let protocol_req = Self::to_protocol_request(req);
        match self.state.worker_pool.process(protocol_req).await {
            Ok(resp) => QueryResponse {
                status: 200, body: resp.body, error: String::new(), error_code: String::new(),
            },
            Err(e) => {
                let status = error_status_code(&e);
                QueryResponse {
                    status: status as i32, body: Vec::new(),
                    error: e.to_string(), error_code: e.code().to_string(),
                }
            }
        }
    }
}

#[tonic::async_trait]
impl HelixDb for HelixGrpcService {
    async fn query(&self, request: Request<QueryRequest>) -> Result<Response<QueryResponse>, Status> {
        let resp = self.execute_one(request.get_ref()).await;
        Ok(Response::new(resp))
    }
    async fn batch(&self, request: Request<BatchRequest>) -> Result<Response<BatchResponse>, Status> {
        let batch = request.into_inner();
        if batch.queries.len() > 50 {
            return Err(Status::invalid_argument("Batch size exceeds maximum of 50 queries"));
        }
        let mut handles = Vec::with_capacity(batch.queries.len());
        for query in &batch.queries {
            let state = Arc::clone(&self.state);
            let protocol_req = Self::to_protocol_request(query);
            handles.push(tokio::spawn(async move {
                match state.worker_pool.process(protocol_req).await {
                    Ok(resp) => QueryResponse {
                        status: 200, body: resp.body, error: String::new(), error_code: String::new(),
                    },
                    Err(e) => {
                        let status = error_status_code(&e);
                        QueryResponse {
                            status: status as i32, body: Vec::new(),
                            error: e.to_string(), error_code: e.code().to_string(),
                        }
                    }
                }
            }));
        }
        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.await {
                Ok(resp) => results.push(resp),
                Err(join_err) => results.push(QueryResponse {
                    status: 500, body: Vec::new(),
                    error: join_err.to_string(), error_code: "INTERNAL_ERROR".to_string(),
                }),
            }
        }
        Ok(Response::new(BatchResponse { results }))
    }
    type StreamQueryStream = ReceiverStream<Result<QueryResponse, Status>>;
    async fn stream_query(&self, request: Request<Streaming<QueryRequest>>) -> Result<Response<Self::StreamQueryStream>, Status> {
        let mut inbound = request.into_inner();
        let state = Arc::clone(&self.state);
        let (tx, rx) = mpsc::channel(64);
        tokio::spawn(async move {
            while let Ok(Some(query_req)) = inbound.message().await {
                let state = Arc::clone(&state);
                let tx = tx.clone();
                tokio::spawn(async move {
                    let protocol_req = HelixGrpcService::to_protocol_request(&query_req);
                    let resp = match state.worker_pool.process(protocol_req).await {
                        Ok(resp) => QueryResponse {
                            status: 200, body: resp.body, error: String::new(), error_code: String::new(),
                        },
                        Err(e) => {
                            let status = error_status_code(&e);
                            QueryResponse {
                                status: status as i32, body: Vec::new(),
                                error: e.to_string(), error_code: e.code().to_string(),
                            }
                        }
                    };
                    if tx.send(Ok(resp)).await.is_err() {
                        warn!("gRPC stream client disconnected");
                    }
                });
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

fn error_status_code(e: &protocol::HelixError) -> u16 {
    use crate::helix_engine::types::{GraphError, VectorError};
    match e {
        protocol::HelixError::NotFound { .. }
        | protocol::HelixError::Graph(
            GraphError::ConfigFileNotFound | GraphError::NodeNotFound
            | GraphError::EdgeNotFound | GraphError::LabelNotFound
            | GraphError::ShortestPathNotFound,
        )
        | protocol::HelixError::Vector(VectorError::VectorNotFound(_)) => 404,
        protocol::HelixError::InvalidApiKey => 403,
        _ => 500,
    }
}
