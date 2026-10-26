//! Reranking of conversation search results, opt-in via `--rerank`: a local
//! cross-encoder by default, or a remote rerank API with `--rerank-remote`.

use crate::machine::LocatedRecord;
use crate::remote_embed::{
    AttemptError, DEFAULT_BACKOFF_BASE, PostOutcome, RemoteEndpoint, build_agent, classify_http,
    parse_body, post_json, retry_transient,
};
use anyhow::{Result, anyhow, bail};
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;
use ureq::Agent;

/// How many fused records rerank by default when `--rerank-limit` is omitted.
pub(crate) const DEFAULT_RERANK_LIMIT: usize = 50;
pub(crate) const MAX_RERANK_LIMIT: usize = 200;

/// Cross-encoder model names accepted by `--rerank-model`.
pub(crate) const RERANK_MODEL_CHOICES: [&str; 4] = ["bge", "bge-m3", "jina", "jina-multilingual"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RerankBackend {
    Local,
    Remote,
}

#[derive(Debug, Clone)]
pub(crate) struct RerankOptions {
    pub limit: usize,
    pub model: Option<String>,
    pub backend: RerankBackend,
}

/// `--rerank` only affects the conversation corpus; memory corpora skip it.
pub(crate) fn validate_rerank_content(
    rerank: bool,
    content: crate::cli::SearchContent,
) -> Result<()> {
    if rerank && content != crate::cli::SearchContent::Conversations {
        return Err(anyhow!("reranking applies to conversation search only"));
    }
    Ok(())
}

/// Parse `--rerank-model` values into fastembed's model enum.
pub(crate) fn parse_rerank_model(value: &str) -> Result<RerankerModel> {
    match value.to_ascii_lowercase().as_str() {
        "bge" | "" => Ok(RerankerModel::BGERerankerBase),
        "bge-m3" => Ok(RerankerModel::BGERerankerV2M3),
        "jina" => Ok(RerankerModel::JINARerankerV1TurboEn),
        "jina-multilingual" => Ok(RerankerModel::JINARerankerV2BaseMultiligual),
        other => Err(anyhow!(
            "unknown rerank model '{other}', options: {}",
            RERANK_MODEL_CHOICES.join(", ")
        )),
    }
}

/// Per-field character budget for tool payloads inside a rerank document.
const TOOL_FIELD_BUDGET: usize = 2_000;

/// Document text fed to the cross-encoder for one record.
fn document_text(record: &crate::types::Record) -> String {
    let mut document = String::with_capacity(record.text.len());
    document.push_str(&record.text);
    for field in [&record.tool_input, &record.tool_output] {
        let Some(field) = field else { continue };
        document.push('\n');
        if field.chars().count() <= TOOL_FIELD_BUDGET {
            document.push_str(field);
        } else {
            let truncated: String = field.chars().take(TOOL_FIELD_BUDGET).collect();
            document.push_str(&truncated);
        }
    }
    document
}

fn documents_for(results: &[LocatedRecord]) -> Vec<String> {
    results
        .iter()
        .map(|located| document_text(&located.record))
        .collect()
}

fn apply_scores(results: &mut [LocatedRecord], scores: Vec<f32>) {
    debug_assert_eq!(scores.len(), results.len());
    for (position, located) in results.iter_mut().enumerate() {
        located.score = scores[position];
    }
    results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

pub(crate) enum Reranker {
    Local(Box<TextRerank>),
    Remote(RemoteReranker),
}

/// One rerank scoring pass over prepared documents, shared by both backends so the
/// pipeline can be exercised without a model.
pub(crate) trait RerankScorer {
    /// One score per document, in input order.
    fn score(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>>;
}

impl RerankScorer for Reranker {
    fn score(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        match self {
            Reranker::Local(inner) => {
                let document_refs: Vec<&str> = documents.iter().map(String::as_str).collect();
                let mut scores: Vec<f32> = vec![0.0; documents.len()];
                for hit in inner.rerank(query, &document_refs, false, None)? {
                    if hit.index >= scores.len() {
                        bail!(
                            "reranker returned out-of-range document index {}",
                            hit.index
                        );
                    }
                    scores[hit.index] = hit.score;
                }
                Ok(scores)
            }
            Reranker::Remote(remote) => {
                let document_refs: Vec<&str> = documents.iter().map(String::as_str).collect();
                remote.fetch_scores(query, &document_refs)
            }
        }
    }
}

/// A rerank pass configured for one search and handed to the shared search pipeline.
///
/// The pipeline reranks the fused candidate pool rather than the rendered page, so the
/// rescoring sees the records retrieval actually ranked highest, not the ones left after
/// the page was truncated.
pub(crate) struct RerankPass {
    scorer: Box<dyn RerankScorer>,
    limit: usize,
}

impl RerankPass {
    pub(crate) fn for_search(
        options: &RerankOptions,
        config: &crate::config::UserConfig,
        cache_dir: &Path,
    ) -> Result<Self> {
        Ok(Self {
            scorer: Box::new(Reranker::for_search(options, config, cache_dir)?),
            limit: options.limit.clamp(1, MAX_RERANK_LIMIT),
        })
    }

    /// Leading fused records this pass rescores, and therefore the pool retrieval must reach.
    pub(crate) fn limit(&self) -> usize {
        self.limit
    }

    pub(crate) fn apply(&mut self, query: &str, results: &mut [LocatedRecord]) -> Result<()> {
        let window_end = self.limit.min(results.len());
        let window = &mut results[..window_end];
        if window.len() < 2 {
            return Ok(());
        }
        let documents = documents_for(window);
        let scores = self.scorer.score(query, &documents)?;
        apply_scores(window, scores);
        Ok(())
    }
}

#[cfg(test)]
impl RerankPass {
    /// A pass over an injected scorer, so pipeline tests need no model download.
    pub(crate) fn with_scorer(scorer: Box<dyn RerankScorer>, limit: usize) -> Self {
        Self {
            scorer,
            limit: limit.clamp(1, MAX_RERANK_LIMIT),
        }
    }
}

impl Reranker {
    /// Resolve only the selected backend's configuration.
    pub(crate) fn for_search(
        options: &RerankOptions,
        config: &crate::config::UserConfig,
        embed_cache_dir: &Path,
    ) -> Result<Self> {
        match options.backend {
            RerankBackend::Local => Self::local(embed_cache_dir, options.model.as_deref()),
            RerankBackend::Remote => {
                let endpoint = config.resolve_remote_rerank()?;
                let model = trimmed(options.model.as_deref())
                    .or_else(|| trimmed(config.rerank_model.as_deref()));
                Ok(Self::Remote(RemoteReranker::new(endpoint, model)?))
            }
        }
    }

    fn local(embed_cache_dir: &Path, model: Option<&str>) -> Result<Self> {
        std::fs::create_dir_all(embed_cache_dir)?;
        let model_name = parse_rerank_model(model.unwrap_or("bge"))?;
        let inner = TextRerank::try_new(
            RerankInitOptions::new(model_name)
                .with_cache_dir(PathBuf::from(embed_cache_dir))
                .with_show_download_progress(false),
        )?;
        Ok(Self::Local(Box::new(inner)))
    }
}

fn trimmed(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

/// Blocking client for a Cohere/Voyage/Jina-style `{base_url}/rerank` endpoint.
pub(crate) struct RemoteReranker {
    endpoint: RemoteEndpoint,
    agent: Agent,
    url: String,
    model: String,
    backoff_base: Duration,
}

impl RemoteReranker {
    pub(crate) fn new(endpoint: RemoteEndpoint, model: Option<&str>) -> Result<Self> {
        let model = trimmed(model).ok_or_else(|| {
            anyhow!("remote reranking requires a model; set rerank_model or pass --rerank-model")
        })?;
        endpoint.validate_for("rerank")?;
        let url = format!("{}/rerank", endpoint.base_url.trim_end_matches('/'));
        let agent = build_agent(endpoint.timeout);
        Ok(Self {
            endpoint,
            agent,
            url,
            model: model.to_owned(),
            backoff_base: DEFAULT_BACKOFF_BASE,
        })
    }

    fn fetch_scores(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        let body = RerankRequest {
            model: &self.model,
            query,
            documents,
        };
        retry_transient(
            &self.url,
            "rerank",
            self.endpoint.max_retries,
            self.backoff_base,
            || self.attempt(&body, documents.len()),
        )
    }

    fn attempt(&self, body: &RerankRequest<'_>, expected: usize) -> Result<Vec<f32>, AttemptError> {
        match post_json(
            &self.agent,
            &self.url,
            self.endpoint.bearer_key(),
            "rerank",
            body,
        )? {
            PostOutcome::Success { body: raw } => {
                let parsed: RerankResponse = parse_body(&raw, "rerank")?;
                validate_scores(parsed, expected).map_err(AttemptError::Fatal)
            }
            PostOutcome::Error {
                status,
                retry_after,
                message,
            } => Err(classify_http(
                status,
                anyhow!("rerank server returned HTTP {status}: {message}"),
                retry_after,
            )),
        }
    }

    #[cfg(test)]
    fn set_backoff_base(&mut self, base: Duration) {
        self.backoff_base = base;
    }
}

#[derive(Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [&'a str],
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Option<Vec<RerankItem>>,
    data: Option<Vec<RerankItem>>,
}

#[derive(Deserialize)]
struct RerankItem {
    index: usize,
    #[serde(alias = "relevance_score")]
    score: f32,
}

fn validate_scores(response: RerankResponse, expected: usize) -> Result<Vec<f32>> {
    let items = response
        .results
        .or(response.data)
        .ok_or_else(|| anyhow!("rerank server response contains neither `results` nor `data`"))?;
    if items.len() != expected {
        bail!(
            "rerank server returned {} results for {expected} documents",
            items.len()
        );
    }
    let mut slots: Vec<Option<f32>> = vec![None; expected];
    for item in items {
        if !item.score.is_finite() {
            bail!(
                "rerank server returned a non-finite score at index {}",
                item.index
            );
        }
        let Some(slot) = slots.get_mut(item.index) else {
            bail!(
                "rerank server returned index {} for {expected} documents",
                item.index
            );
        };
        if slot.is_some() {
            bail!("rerank server returned index {} twice", item.index);
        }
        *slot = Some(item.score);
    }
    // Count, uniqueness, and range make the mapping a bijection.
    Ok(slots
        .into_iter()
        .map(|slot| slot.expect("complete bijection"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Record, RecordLinks, SourceKind};
    use serde_json::{Value, json};

    fn record(text: &str) -> Record {
        Record {
            source: SourceKind::Codex,
            doc_id: 0,
            ts: 0,
            project: "project".to_string(),
            session_id: "session".to_string(),
            turn_id: 0,
            role: "assistant".to_string(),
            text: text.to_string(),
            tool_name: None,
            tool_input: None,
            tool_output: None,
            links: RecordLinks::default(),
            source_path: "/tmp/session.jsonl".to_string(),
        }
    }

    fn texts(results: &[LocatedRecord]) -> Vec<&str> {
        results
            .iter()
            .map(|located| located.record.text.as_str())
            .collect()
    }

    fn scores(results: &[LocatedRecord]) -> Vec<f32> {
        results.iter().map(|located| located.score).collect()
    }

    #[test]
    fn apply_scores_flips_two_records_into_descending_order() {
        let mut results = vec![located("first", 0.8), located("second", 0.7)];
        apply_scores(&mut results, vec![0.1, 0.9]);
        assert_eq!(texts(&results), vec!["second", "first"]);
        assert_eq!(scores(&results), vec![0.9, 0.1]);
    }

    #[test]
    fn apply_scores_permutes_three_records_without_losing_any() {
        let mut results = vec![
            located("first", 0.8),
            located("second", 0.7),
            located("third", 0.6),
        ];
        apply_scores(&mut results, vec![0.5, 0.1, 0.9]);
        assert_eq!(texts(&results), vec!["third", "first", "second"]);
        assert_eq!(scores(&results), vec![0.9, 0.5, 0.1]);
    }

    #[test]
    fn apply_scores_keeps_input_rank_for_ties_and_nan() {
        // Equal scores keep the pre-rerank rank.
        let mut results = vec![
            located("first", 0.8),
            located("second", 0.7),
            located("third", 0.6),
        ];
        apply_scores(&mut results, vec![0.4, 0.4, 0.9]);
        assert_eq!(texts(&results), vec!["third", "first", "second"]);

        // NaN compares Equal to everything, so it holds its input position.
        let mut results = vec![located("first", 0.8), located("second", 0.7)];
        apply_scores(&mut results, vec![f32::NAN, 1.0]);
        assert_eq!(texts(&results), vec!["first", "second"]);
    }

    #[test]
    fn model_parse_matches_documented_names() {
        assert!(matches!(
            parse_rerank_model("bge"),
            Ok(RerankerModel::BGERerankerBase)
        ));
        assert!(matches!(
            parse_rerank_model("BGE-M3"),
            Ok(RerankerModel::BGERerankerV2M3)
        ));
        assert!(matches!(
            parse_rerank_model("jina"),
            Ok(RerankerModel::JINARerankerV1TurboEn)
        ));
        assert!(matches!(
            parse_rerank_model("jina-multilingual"),
            Ok(RerankerModel::JINARerankerV2BaseMultiligual)
        ));
        assert!(parse_rerank_model("gemma").is_err());
    }

    #[test]
    fn rerank_guard_rejects_memory_content() {
        assert!(validate_rerank_content(true, crate::cli::SearchContent::Conversations).is_ok());
        assert!(validate_rerank_content(false, crate::cli::SearchContent::Memories).is_ok());
        let error = validate_rerank_content(true, crate::cli::SearchContent::Memories)
            .expect_err("memories + rerank must fail");
        assert_eq!(
            error.to_string(),
            "reranking applies to conversation search only"
        );
    }

    #[test]
    fn document_text_appends_truncated_tool_fields() {
        let mut record = record("answer");
        record.tool_input = Some("input".to_string());
        record.tool_output = Some("x".repeat(TOOL_FIELD_BUDGET + 10));
        let document = document_text(&record);
        assert!(document.starts_with("answer\ninput\n"));
        assert_eq!(
            document.chars().count(),
            "answer\ninput\n".len() + TOOL_FIELD_BUDGET
        );
    }

    // --- remote backend ---

    use std::thread::{self, JoinHandle};
    use tiny_http::{Header, Response, Server};

    const KEY: &str = "sk-rerank-SECRET-7f3a9c41";

    /// Run one rerank pass over `results` exactly as the search pipeline does.
    fn rerank_pass(
        reranker: Reranker,
        limit: usize,
        query: &str,
        results: &mut [LocatedRecord],
    ) -> Result<()> {
        RerankPass::with_scorer(Box::new(reranker), limit).apply(query, results)
    }

    fn located(text: &str, score: f32) -> LocatedRecord {
        LocatedRecord {
            machine: "local".to_string(),
            score,
            record: record(text),
        }
    }

    struct Captured {
        path: String,
        auth: Option<String>,
        body: Value,
    }

    /// Serves one canned reply per request until 500ms pass without one.
    fn serve<F>(mut handler: F) -> Result<(String, JoinHandle<Vec<Captured>>)>
    where
        F: FnMut(usize, &Value) -> (u16, Value) + Send + 'static,
    {
        let server = Server::http("127.0.0.1:0").map_err(|e| anyhow!("bind: {e}"))?;
        let addr = server
            .server_addr()
            .to_ip()
            .ok_or_else(|| anyhow!("no ip address"))?;
        let handle = thread::spawn(move || {
            let mut captured = Vec::new();
            while let Ok(Some(mut request)) = server.recv_timeout(Duration::from_millis(500)) {
                let mut raw = String::new();
                let _ = request.as_reader().read_to_string(&mut raw);
                let body: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
                let auth = request
                    .headers()
                    .iter()
                    .find(|header| header.field.equiv("Authorization"))
                    .map(|header| header.value.as_str().to_owned());
                let (status, reply) = handler(captured.len(), &body);
                let mut response =
                    Response::from_string(reply.to_string()).with_status_code(status);
                if let Ok(header) = Header::from_bytes("Content-Type", "application/json") {
                    response = response.with_header(header);
                }
                captured.push(Captured {
                    path: request.url().to_owned(),
                    auth,
                    body,
                });
                let _ = request.respond(response);
            }
            captured
        });
        Ok((format!("http://{addr}/v1"), handle))
    }

    fn join(handle: JoinHandle<Vec<Captured>>) -> Result<Vec<Captured>> {
        handle.join().map_err(|_| anyhow!("server thread panicked"))
    }

    fn expect_err<T>(result: Result<T>) -> Result<String> {
        match result {
            Ok(_) => bail!("expected an error"),
            Err(error) => Ok(format!("{error:#}")),
        }
    }

    fn endpoint(base: &str, key: Option<&str>, max_retries: u32) -> RemoteEndpoint {
        RemoteEndpoint {
            base_url: base.to_owned(),
            api_key: key.map(str::to_owned),
            timeout: Duration::from_secs(5),
            max_retries,
        }
    }

    fn remote(base: &str, key: Option<&str>, max_retries: u32) -> Result<RemoteReranker> {
        let mut reranker =
            RemoteReranker::new(endpoint(base, key, max_retries), Some("test-model"))?;
        reranker.set_backoff_base(Duration::from_millis(1));
        Ok(reranker)
    }

    #[test]
    fn config_dispatch_sends_auth_and_payload_and_aligns_scores() -> Result<()> {
        let (base, server) = serve(|_, _| {
            (
                200,
                json!({"results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 1, "relevance_score": 0.1}
                ]}),
            )
        })?;
        let config: crate::config::UserConfig = toml::from_str(&format!(
            "rerank_base_url = \"{base}\"\nrerank_model = \"provider-model\"\nrerank_api_key = \"{KEY}\""
        ))
        .map_err(|e| anyhow!("toml: {e}"))?;
        let options = RerankOptions {
            limit: 50,
            model: None,
            backend: RerankBackend::Remote,
        };
        let reranker = Reranker::for_search(&options, &config, Path::new("unused-embed-cache"))?;
        let mut results = vec![
            located("first", 0.8),
            located("second", 0.7),
            located("third", 0.6),
        ];
        rerank_pass(reranker, 50, "needle", &mut results)?;
        let captured = join(server)?;
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].path, "/v1/rerank");
        let auth = format!("Bearer {KEY}");
        assert_eq!(captured[0].auth.as_deref(), Some(auth.as_str()));
        assert_eq!(captured[0].body["model"], json!("provider-model"));
        assert_eq!(captured[0].body["query"], json!("needle"));
        assert_eq!(
            captured[0].body["documents"],
            json!(["first", "second", "third"])
        );
        // Out-of-order response: scores align to the original inputs.
        let texts: Vec<&str> = results
            .iter()
            .map(|located| located.record.text.as_str())
            .collect();
        assert_eq!(texts, vec!["third", "first", "second"]);
        assert_eq!(results[0].score, 0.9);
        assert_eq!(results[1].score, 0.5);
        assert_eq!(results[2].score, 0.1);
        Ok(())
    }

    #[test]
    fn accepts_data_score_response_shape() -> Result<()> {
        let (base, server) = serve(|_, _| {
            (
                200,
                json!({"data": [
                    {"index": 0, "score": 0.2},
                    {"index": 1, "score": 0.8}
                ]}),
            )
        })?;
        let reranker = Reranker::Remote(remote(&base, None, 0)?);
        let mut results = vec![located("a", 0.1), located("b", 0.1)];
        rerank_pass(reranker, 50, "q", &mut results)?;
        join(server)?;
        let texts: Vec<&str> = results
            .iter()
            .map(|located| located.record.text.as_str())
            .collect();
        assert_eq!(texts, vec!["b", "a"]);
        assert_eq!(results[0].score, 0.8);
        assert_eq!(results[1].score, 0.2);
        Ok(())
    }

    #[test]
    fn rejects_malformed_scores() -> Result<()> {
        let three = |index_zero: Value, index_one: Value, index_two: Value| json!({"results": [index_zero, index_one, index_two]});
        let cases: [(&str, Value, &str); 4] = [
            (
                "count",
                json!({"results": [
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 1, "relevance_score": 0.4}
                ]}),
                "returned 2 results for 3 documents",
            ),
            (
                "duplicate",
                three(
                    json!({"index": 0, "relevance_score": 0.5}),
                    json!({"index": 0, "relevance_score": 0.4}),
                    json!({"index": 2, "relevance_score": 0.1}),
                ),
                "index 0 twice",
            ),
            (
                "range",
                three(
                    json!({"index": 0, "relevance_score": 0.5}),
                    json!({"index": 1, "relevance_score": 0.4}),
                    json!({"index": 3, "relevance_score": 0.1}),
                ),
                "index 3 for 3 documents",
            ),
            (
                // Upstream's precise f32 parsing rejects overflow while deserializing, so
                // this never reaches the score validator.
                "overflow",
                three(
                    json!({"index": 0, "relevance_score": 1e39}),
                    json!({"index": 1, "relevance_score": 0.4}),
                    json!({"index": 2, "relevance_score": 0.1}),
                ),
                "number out of range",
            ),
        ];
        for (name, reply, expected) in cases {
            let (base, server) = serve(move |_, _| (200, reply.clone()))?;
            let reranker = Reranker::Remote(remote(&base, None, 0)?);
            let mut results = vec![located("a", 1.0), located("b", 1.0), located("c", 1.0)];
            let error = expect_err(rerank_pass(reranker, 50, "q", &mut results))?;
            join(server)?;
            assert!(error.contains(expected), "{name}: {error}");
        }
        Ok(())
    }

    #[test]
    fn failure_is_single_attempt_and_redacts_key() -> Result<()> {
        let (base, server) = serve(|_, _| {
            (
                401,
                json!({"error": {"message": format!("Incorrect API key provided: {KEY}")}}),
            )
        })?;
        let reranker = Reranker::Remote(remote(&base, Some(KEY), 0)?);
        let mut results = vec![located("a", 1.0), located("b", 1.0)];
        let error = expect_err(rerank_pass(reranker, 50, "q", &mut results))?;
        let captured = join(server)?;
        assert_eq!(captured.len(), 1, "zero retries means one attempt");
        assert!(error.contains("HTTP 401"), "{error}");
        assert!(!error.contains(KEY), "{error}");
        assert!(error.contains("<redacted>"), "{error}");
        Ok(())
    }

    #[test]
    fn retries_transient_failures_when_configured() -> Result<()> {
        let (base, server) = serve(|attempt, _| match attempt {
            0 => (503, json!({"error": {"message": "overloaded"}})),
            _ => (
                200,
                json!({"results": [
                    {"index": 0, "relevance_score": 0.4},
                    {"index": 1, "relevance_score": 0.6}
                ]}),
            ),
        })?;
        let reranker = Reranker::Remote(remote(&base, None, 1)?);
        let mut results = vec![located("a", 1.0), located("b", 1.0)];
        rerank_pass(reranker, 50, "q", &mut results)?;
        assert_eq!(join(server)?.len(), 2);
        assert_eq!(results[0].record.text, "b");
        Ok(())
    }

    #[test]
    fn remote_model_resolves_cli_then_config_or_fails() -> Result<()> {
        // Construction never sends a request, so an unreachable base URL is fine.
        let base = "http://127.0.0.1:9";
        let config = |model: Option<&str>| -> crate::config::UserConfig {
            let model = model
                .map(|model| format!("rerank_model = \"{model}\"\n"))
                .unwrap_or_default();
            toml::from_str(&format!("rerank_base_url = \"{base}\"\n{model}"))
                .expect("parse rerank config")
        };
        let remote_options = |model: Option<&str>| RerankOptions {
            limit: 50,
            model: model.map(str::to_owned),
            backend: RerankBackend::Remote,
        };

        let Reranker::Remote(reranker) = Reranker::for_search(
            &remote_options(None),
            &config(Some("cfg-model")),
            Path::new("unused-embed-cache"),
        )?
        else {
            panic!("expected the remote backend");
        };
        assert_eq!(reranker.model, "cfg-model");

        let Reranker::Remote(reranker) = Reranker::for_search(
            &remote_options(Some("cli-model")),
            &config(Some("cfg-model")),
            Path::new("unused-embed-cache"),
        )?
        else {
            panic!("expected the remote backend");
        };
        assert_eq!(reranker.model, "cli-model");

        let error = expect_err(Reranker::for_search(
            &remote_options(None),
            &config(None),
            Path::new("unused-embed-cache"),
        ))?;
        assert!(
            error.contains("remote reranking requires a model"),
            "{error}"
        );

        let error = expect_err(Reranker::for_search(
            &remote_options(Some("cli-model")),
            &crate::config::UserConfig::default(),
            Path::new("unused-embed-cache"),
        ))?;
        assert!(error.contains("rerank_base_url"), "{error}");
        Ok(())
    }
}
