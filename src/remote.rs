//! OpenAI-compatible remote embedding and reranking endpoints.

use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::time::Duration;

/// Per-request HTTP timeout. Remote search is a slow path; failing fast keeps
/// `memex search` responsive enough to fall back to local behavior.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Batch size for `/v1/embeddings` calls during bulk indexing.
const EMBED_BATCH: usize = 96;

/// Endpoint configuration resolved from config file and environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteConfig {
    /// Base URL ending at `/v1` (e.g. `https://api.voyageai.com/v1`).
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub dimensions: Option<usize>,
    /// Send `input_type` (`query`/`document`) — Voyage and Jina accept it,
    /// OpenAI-compatible servers that mirror only the core schema reject it.
    pub input_type: bool,
}

impl RemoteConfig {
    /// Resolve from environment: `MEMEX_EMBEDDINGS_ENDPOINT`, `_API_KEY`,
    /// `_MODEL`, `_DIMENSIONS`, `_INPUT_TYPE`. Returns `None` when no endpoint
    /// is configured so callers can fall back to local embedders.
    pub fn from_env() -> Result<Option<Self>> {
        let endpoint = match std::env::var("MEMEX_EMBEDDINGS_ENDPOINT") {
            Ok(value) if !value.trim().is_empty() => value.trim().trim_end_matches('/').to_string(),
            _ => return Ok(None),
        };
        let api_key = std::env::var("MEMEX_EMBEDDINGS_API_KEY").unwrap_or_default();
        let model = std::env::var("MEMEX_EMBEDDINGS_MODEL").unwrap_or_default();
        if model.trim().is_empty() {
            return Err(anyhow!(
                "MEMEX_EMBEDDINGS_ENDPOINT is set but MEMEX_EMBEDDINGS_MODEL is empty"
            ));
        }
        let dimensions = match std::env::var("MEMEX_EMBEDDINGS_DIMENSIONS") {
            Ok(value) if !value.trim().is_empty() => Some(value.trim().parse::<usize>()?),
            _ => None,
        };
        let input_type = std::env::var("MEMEX_EMBEDDINGS_INPUT_TYPE")
            .map(|value| !matches!(value.trim(), "0" | "false" | "no" | ""))
            .unwrap_or(true);
        Ok(Some(Self {
            endpoint,
            api_key,
            model,
            dimensions,
            input_type,
        }))
    }
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Blocking client for an OpenAI-compatible `/v1/embeddings` endpoint.
pub struct RemoteEmbedder {
    config: RemoteConfig,
    client: reqwest::blocking::Client,
    dims: usize,
}

impl RemoteEmbedder {
    pub fn new(config: RemoteConfig) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()?;
        let dims = config.dimensions.ok_or_else(|| {
            anyhow!(
                "remote embeddings require MEMEX_EMBEDDINGS_DIMENSIONS (the API does not report \
                 output size before the first call)"
            )
        })?;
        Ok(Self {
            config,
            client,
            dims,
        })
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn embed(&self, texts: &[&str], input_type: Option<&str>) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(EMBED_BATCH) {
            vectors.extend(self.embed_batch(batch, input_type)?);
        }
        Ok(vectors)
    }

    fn embed_batch(&self, batch: &[&str], input_type: Option<&str>) -> Result<Vec<Vec<f32>>> {
        let mut body = serde_json::json!({
            "input": batch,
            "model": self.config.model,
        });
        if let Some(input_type) = input_type {
            body["input_type"] = serde_json::Value::String(input_type.to_string());
        }
        if let Some(dimensions) = self.config.dimensions {
            body["output_dimension"] = serde_json::json!(dimensions);
        }
        let response: EmbeddingsResponse = self.post("/embeddings", body)?;
        if response.data.len() != batch.len() {
            return Err(anyhow!(
                "remote embeddings returned {} vectors for {} inputs",
                response.data.len(),
                batch.len()
            ));
        }
        let vectors: Vec<Vec<f32>> = response
            .data
            .into_iter()
            .map(|item| item.embedding)
            .collect();
        let expected = self.dims;
        if let Some(bad) = vectors.iter().find(|vector| vector.len() != expected) {
            return Err(anyhow!(
                "remote embeddings returned {} dims, expected {expected}",
                bad.len()
            ));
        }
        Ok(vectors)
    }

    fn post<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> Result<T> {
        let mut request = self
            .client
            .post(format!("{}{path}", self.config.endpoint))
            .json(&body);
        if !self.config.api_key.is_empty() {
            request = request.bearer_auth(&self.config.api_key);
        }
        let response = request.send()?;
        let status = response.status();
        let payload = response.text()?;
        if !status.is_success() {
            let snippet: String = payload.chars().take(300).collect();
            return Err(anyhow!("remote embeddings HTTP {status}: {snippet}"));
        }
        Ok(serde_json::from_str(&payload)?)
    }
}

#[derive(Deserialize)]
struct RerankResponse {
    data: Vec<RerankHit>,
}

#[derive(Deserialize)]
struct RerankHit {
    index: usize,
    #[serde(rename = "relevance_score")]
    score: f32,
}

/// Blocking client for a Voyage/Jina-style `/v1/rerank` endpoint.
pub struct RemoteReranker {
    config: RemoteConfig,
    model: String,
    client: reqwest::blocking::Client,
}

impl RemoteReranker {
    pub fn new(config: RemoteConfig, model: Option<&str>) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()?;
        // Remote rerank models are endpoint-specific (voyage: rerank-3*,
        // jina: jina-reranker-v3), so the model always comes from
        // --rerank-model / MEMEX_RERANKER_MODEL, never the embedding model.
        let model = match model.map(str::trim).filter(|value| !value.is_empty()) {
            Some(model) => model.to_string(),
            None => std::env::var("MEMEX_RERANKER_MODEL")
                .unwrap_or_else(|_| "rerank-3-lite".to_string()),
        };
        Ok(Self {
            config,
            model,
            client,
        })
    }

    /// The resolved remote rerank model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns scores aligned with the input `documents`.
    pub fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        let body = serde_json::json!({
            "query": query,
            "documents": documents,
            "model": self.model,
        });
        let mut request = self
            .client
            .post(format!("{}/rerank", self.config.endpoint))
            .json(&body);
        if !self.config.api_key.is_empty() {
            request = request.bearer_auth(&self.config.api_key);
        }
        let response = request.send()?;
        let status = response.status();
        let payload = response.text()?;
        if !status.is_success() {
            let snippet: String = payload.chars().take(300).collect();
            return Err(anyhow!("remote rerank HTTP {status}: {snippet}"));
        }
        let parsed: RerankResponse = serde_json::from_str(&payload)?;
        let mut scores = vec![0.0_f32; documents.len()];
        for hit in parsed.data {
            if hit.index >= scores.len() {
                return Err(anyhow!(
                    "remote rerank returned out-of-range document index {}",
                    hit.index
                ));
            }
            scores[hit.index] = hit.score;
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{EnvVarGuard, env_lock};

    fn endpoint_guard() -> EnvVarGuard {
        EnvVarGuard::set(&[
            ("MEMEX_EMBEDDINGS_ENDPOINT", None),
            ("MEMEX_EMBEDDINGS_MODEL", None),
            ("MEMEX_EMBEDDINGS_API_KEY", None),
            ("MEMEX_EMBEDDINGS_DIMENSIONS", None),
            ("MEMEX_EMBEDDINGS_INPUT_TYPE", None),
        ])
    }

    #[test]
    fn remote_config_requires_endpoint() {
        let _lock = env_lock();
        let _guard = endpoint_guard();
        assert!(RemoteConfig::from_env().unwrap().is_none());
    }

    #[test]
    fn remote_config_parses_env_block() {
        let _lock = env_lock();
        let _guard = endpoint_guard();
        let _set = EnvVarGuard::set(&[
            (
                "MEMEX_EMBEDDINGS_ENDPOINT",
                Some("https://api.voyageai.com/v1/"),
            ),
            ("MEMEX_EMBEDDINGS_MODEL", Some("voyage-4-lite")),
            ("MEMEX_EMBEDDINGS_API_KEY", Some("secret")),
            ("MEMEX_EMBEDDINGS_DIMENSIONS", Some("1024")),
        ]);
        let config = RemoteConfig::from_env().unwrap().expect("config");
        assert_eq!(config.endpoint, "https://api.voyageai.com/v1");
        assert_eq!(config.model, "voyage-4-lite");
        assert_eq!(config.api_key, "secret");
        assert_eq!(config.dimensions, Some(1024));
        assert!(config.input_type);
    }

    #[test]
    fn remote_config_rejects_missing_model() {
        let _lock = env_lock();
        let _guard = endpoint_guard();
        let _set = EnvVarGuard::set(&[
            ("MEMEX_EMBEDDINGS_ENDPOINT", Some("https://example.com/v1")),
            ("MEMEX_EMBEDDINGS_MODEL", None),
        ]);
        assert!(RemoteConfig::from_env().is_err());
    }

    #[test]
    fn remote_config_input_type_off_switch() {
        let _lock = env_lock();
        let _guard = endpoint_guard();
        let _set = EnvVarGuard::set(&[
            ("MEMEX_EMBEDDINGS_ENDPOINT", Some("https://example.com/v1")),
            ("MEMEX_EMBEDDINGS_MODEL", Some("text-embedding-3-small")),
            ("MEMEX_EMBEDDINGS_INPUT_TYPE", Some("0")),
        ]);
        let config = RemoteConfig::from_env().unwrap().expect("config");
        assert!(!config.input_type);
    }

    #[test]
    fn remote_embedder_requires_dimensions() {
        let config = RemoteConfig {
            endpoint: "https://example.com/v1".into(),
            api_key: String::new(),
            model: "voyage-4-lite".into(),
            dimensions: None,
            input_type: true,
        };
        assert!(RemoteEmbedder::new(config).is_err());
    }
}
