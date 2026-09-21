//! Cross-encoder reranking of conversation search results, opt-in via `--rerank`.

use crate::machine::LocatedRecord;
use anyhow::{Result, anyhow};
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use std::path::{Path, PathBuf};

/// How many fused records rerank by default when `--rerank-limit` is omitted.
pub(crate) const DEFAULT_RERANK_LIMIT: usize = 50;
pub(crate) const MAX_RERANK_LIMIT: usize = 200;

/// Cross-encoder model names accepted by `--rerank-model`.
pub(crate) const RERANK_MODEL_CHOICES: [&str; 4] = ["bge", "bge-m3", "jina", "jina-multilingual"];

#[derive(Debug, Clone)]
pub(crate) struct RerankOptions {
    pub limit: usize,
    pub model: Option<String>,
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

/// Orders records by descending score with a stable tiebreak on the pre-rerank rank.
fn order_by_score(scores: Vec<f32>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|left, right| {
        scores[*right]
            .partial_cmp(&scores[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    order
}

pub(crate) struct Reranker {
    inner: TextRerank,
}

impl Reranker {
    pub(crate) fn new(embed_cache_dir: &Path, model: Option<&str>) -> Result<Self> {
        std::fs::create_dir_all(embed_cache_dir)?;
        let model_name = parse_rerank_model(model.unwrap_or("bge"))?;
        let inner = TextRerank::try_new(
            RerankInitOptions::new(model_name)
                .with_cache_dir(PathBuf::from(embed_cache_dir))
                .with_show_download_progress(false),
        )?;
        Ok(Self { inner })
    }

    /// Rescore `results` in place against `query`; preserves order on ties.
    pub(crate) fn rerank_results(
        &mut self,
        query: &str,
        results: &mut [LocatedRecord],
    ) -> Result<()> {
        if results.len() < 2 {
            return Ok(());
        }
        let documents: Vec<String> = results
            .iter()
            .map(|located| document_text(&located.record))
            .collect();
        let document_refs: Vec<&str> = documents.iter().map(String::as_str).collect();
        let mut scores: Vec<f32> = vec![0.0; documents.len()];
        for hit in self.inner.rerank(query, &document_refs, false, None)? {
            if hit.index >= scores.len() {
                return Err(anyhow!(
                    "reranker returned out-of-range document index {}",
                    hit.index
                ));
            }
            scores[hit.index] = hit.score;
        }
        for (position, located) in results.iter_mut().enumerate() {
            located.score = scores[position];
        }
        for (position, index) in order_by_score(scores).into_iter().enumerate() {
            results.swap(position, index);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Record, RecordLinks, SourceKind};

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

    #[test]
    fn ordering_sorts_descending_with_stable_ties() {
        let order = order_by_score(vec![1.0, 3.0, 3.0, 2.0]);
        assert_eq!(order, vec![1, 2, 3, 0]);
    }

    #[test]
    fn ordering_survives_nan_scores() {
        // NaN compares Equal to everything, so the stable tiebreak keeps the
        // earlier rank first and the real score wins against it.
        let order = order_by_score(vec![f32::NAN, 1.0]);
        assert_eq!(order, vec![0, 1]);
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
}
