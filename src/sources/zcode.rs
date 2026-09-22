//! ZCode session discovery and indexing.
//!
//! ZCode keeps every workspace session in one SQLite store at
//! `~/.zcode/cli/db/db.sqlite`, whether the desktop app runs locally or an
//! SSH-attached agent runtime writes it on a remote host. Sessions live in
//! `session` (with `parent_id` linking subagent sessions to their parent),
//! turns in `message` (JSON `data` carrying role, model, and tokens), and
//! content items in `part`, discriminated by `data.type`: `text`, `reasoning`,
//! and `tool` (input and output in one part), plus skippable `step-start`,
//! `step-finish`, and `timeline` markers. Timestamps are millisecond epochs.
//!
//! The store grows in place as sessions progress. Virtual source paths and
//! content fingerprints let ingestion replace only changed sessions.
//! Token usage comes from the `model_usage` table,
//! one row per model request with its own token buckets.

use super::{
    ConversationKind, IndexParseOutput, IndexParseState, ParseDiagnostics, ParserVersions,
    SourceFile, UsageDependency, UsageParseOutput,
};
use crate::types::{Record, RecordLinks, SourceKind};
use crate::usage::{TokenBuckets, UsageEvent};
use anyhow::{Context, Result, anyhow};
use rusqlite::{Connection, OpenFlags};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub const VERSIONS: ParserVersions = ParserVersions {
    identity: 1,
    index: 2,
    usage: 2,
};

pub fn matches_path(path: &str) -> bool {
    split_virtual_path(Path::new(path)).is_some()
        || path.ends_with("/.zcode/cli/db/db.sqlite")
        || path.ends_with("\\.zcode\\cli\\db\\db.sqlite")
}

/// Encode session ids into a single safe component, including slashes and dots.
pub fn virtual_path(database: &Path, session_id: &str) -> PathBuf {
    let encoded: String = session_id
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    database.join(format!("zcode-session-{encoded}"))
}

pub fn split_virtual_path(path: &Path) -> Option<(PathBuf, String)> {
    let encoded = path.file_name()?.to_str()?.strip_prefix("zcode-session-")?;
    let database = path.parent()?;
    if database.file_name()? != "db.sqlite" || !encoded.len().is_multiple_of(2) {
        return None;
    }
    let bytes = encoded
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok())
        .collect::<Option<Vec<_>>>()?;
    Some((database.to_path_buf(), String::from_utf8(bytes).ok()?))
}

pub(crate) struct SessionFingerprint {
    pub id: String,
    pub fingerprint: String,
    /// Number of joined message/part rows, also returned as the parser offset.
    pub size: u64,
}

/// Hash actual content from one SQLite snapshot; usage-only writes do not
/// invalidate conversation records, and edits need not update a timestamp.
pub(crate) fn enumerate_sessions(database: &Path) -> Result<Vec<SessionFingerprint>> {
    let conn = open_readonly(database)?;
    let transaction = conn.unchecked_transaction()?;
    let mut sessions = transaction.prepare("SELECT id FROM session ORDER BY id")?;
    let ids = sessions
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    drop(sessions);
    let mut result = Vec::with_capacity(ids.len());
    for id in ids {
        let mut hash = Sha256::new();
        for sql in [
            "SELECT * FROM session WHERE id = ?1",
            "SELECT * FROM message WHERE session_id = ?1 ORDER BY sequence, id",
            "SELECT p.* FROM message m JOIN part p ON p.message_id = m.id WHERE m.session_id = ?1 ORDER BY m.sequence, m.id, p.sequence, p.id",
        ] {
            hash.update(sql.as_bytes());
            let mut statement = transaction.prepare(sql)?;
            let columns = statement.column_count();
            let mut rows = statement.query([&id])?;
            while let Some(row) = rows.next()? {
                hash.update([0xff]);
                for column in 0..columns {
                    use rusqlite::types::ValueRef;
                    match row.get_ref(column)? {
                        ValueRef::Null => hash.update([0]),
                        ValueRef::Integer(value) => {
                            hash.update([1]);
                            hash.update(value.to_le_bytes());
                        }
                        ValueRef::Real(value) => {
                            hash.update([2]);
                            hash.update(value.to_le_bytes());
                        }
                        ValueRef::Text(value) | ValueRef::Blob(value) => {
                            hash.update([if matches!(row.get_ref(column)?, ValueRef::Text(_)) {
                                3
                            } else {
                                4
                            }]);
                            hash.update((value.len() as u64).to_le_bytes());
                            hash.update(value);
                        }
                    }
                }
            }
        }
        let size = transaction.query_row(
            "SELECT count(*) FROM message m JOIN part p ON p.message_id = m.id WHERE m.session_id = ?1",
            [&id], |row| row.get::<_, u64>(0),
        )?;
        result.push(SessionFingerprint {
            id,
            fingerprint: format!("{:x}", hash.finalize()),
            size,
        });
    }
    transaction.commit()?;
    Ok(result)
}

/// ZCode state roots: `$ZCODE_HOME` (comma-separated, replacing the default)
/// or `~/.zcode`. Extra roots keep a synced copy of another machine's store
/// indexable next to the live one.
pub fn roots() -> Vec<PathBuf> {
    roots_for(
        std::env::var_os("ZCODE_HOME").as_deref(),
        &super::common::home(),
    )
}

fn roots_for(zcode_home: Option<&std::ffi::OsStr>, home: &Path) -> Vec<PathBuf> {
    match zcode_home {
        Some(roots) => roots
            .to_string_lossy()
            .split(',')
            .filter_map(|root| {
                let root = root.trim();
                (!root.is_empty()).then(|| PathBuf::from(root))
            })
            .collect(),
        None => vec![home.join(".zcode")],
    }
}

/// The session database under each root, when it exists.
pub fn db_paths() -> Vec<PathBuf> {
    db_paths_for(&roots())
}

fn db_paths_for(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = roots
        .iter()
        .map(|root| root.join("cli").join("db").join("db.sqlite"))
        .filter(|path| path.is_file())
        .collect();
    paths.sort();
    paths.dedup();
    paths
}

pub fn discover() -> Vec<SourceFile> {
    discover_from_roots(&roots())
}

pub fn discover_from_roots(roots: &[PathBuf]) -> Vec<SourceFile> {
    db_paths_for(roots)
        .into_iter()
        .map(|path| SourceFile {
            source: SourceKind::Zcode,
            path,
        })
        .collect()
}

pub fn usage_files() -> Vec<PathBuf> {
    db_paths()
}

/// Directories holding the session databases: the watcher's narrowest ZCode
/// roots, clear of the server runtime's churn elsewhere under `~/.zcode`.
pub fn db_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = roots()
        .into_iter()
        .map(|root| root.join("cli").join("db"))
        .filter(|dir| dir.is_dir())
        .collect();
    dirs.sort();
    dirs.dedup();
    dirs
}

fn open_readonly(path: &Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("open ZCode database {}", path.display()))?;
    // The live store is written while we read; never block it or race it long.
    let _ = conn.busy_timeout(std::time::Duration::from_millis(500));
    Ok(conn)
}

fn table_names(conn: &Connection) -> Result<std::collections::HashSet<String>> {
    let mut statement = conn.prepare("SELECT name FROM sqlite_master WHERE type='table'")?;
    Ok(statement
        .query_map([], |row| row.get::<_, String>(0))?
        .filter_map(Result::ok)
        .collect())
}

struct SessionRow {
    id: String,
    parent_id: Option<String>,
    directory: Option<String>,
    time_created: i64,
}

pub(crate) fn parse_index_records(
    path: &Path,
    state: IndexParseState,
    include_reasoning: bool,
    next_doc_id: &AtomicU64,
    mut emit: impl FnMut(Record) -> Result<()>,
) -> Result<IndexParseOutput> {
    // Each changed session is replaced, so state.offset is intentionally ignored.
    let source_path = path.to_string_lossy().to_string();
    let virtual_session = split_virtual_path(path);
    let database = virtual_session
        .as_ref()
        .map_or(path, |(database, _)| database.as_path());
    let selected_session = virtual_session.as_ref().map(|(_, id)| id.as_str());
    let mut diagnostics = ParseDiagnostics::default();
    let conn = open_readonly(database)?;
    // One snapshot across sessions, messages, and parts: in WAL mode ZCode
    // commits rows while we read.
    let transaction = conn.unchecked_transaction()?;
    let tables = table_names(&transaction)?;
    if !(tables.contains("session") && tables.contains("message") && tables.contains("part")) {
        return Err(anyhow!(
            "ZCode database {} has no session/message/part tables",
            path.display()
        ));
    }
    let mut sessions_statement = transaction
        .prepare(
            "SELECT id, parent_id, directory, time_created FROM session WHERE (?1 IS NULL OR id = ?1) ORDER BY time_created, id",
        )
        .with_context(|| format!("query sessions in {}", path.display()))?;
    let sessions = sessions_statement
        .query_map([selected_session], |row| {
            Ok(SessionRow {
                id: row.get(0)?,
                parent_id: row.get(1)?,
                directory: row.get(2)?,
                time_created: row.get::<_, Option<i64>>(3)?.unwrap_or_default(),
            })
        })
        .with_context(|| format!("read sessions in {}", path.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let mut messages_statement = transaction
        .prepare(
            "SELECT m.id, m.sequence, m.time_created, m.data, \
             p.id, p.sequence, p.time_created, p.data \
             FROM message m JOIN part p ON p.message_id = m.id \
             WHERE m.session_id = ?1 ORDER BY m.sequence, m.id, p.sequence, p.id",
        )
        .with_context(|| format!("prepare messages in {}", path.display()))?;

    let mut row_count = 0;
    let mut turn_id = state.turn_id;
    let mut session_cwd: Option<String> = None;
    for session in &sessions {
        let project = session
            .directory
            .as_deref()
            .map(Path::new)
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .map(str::to_string);
        if session_cwd.is_none()
            && let Some(directory) = session.directory.as_deref()
        {
            session_cwd = Some(directory.to_string());
        }
        let rows = messages_statement
            .query_map([session.id.as_str()], |row| {
                Ok((
                    row.get::<_, i64>(1)?,                                // message sequence
                    row.get::<_, Option<i64>>(2)?.unwrap_or_default(),    // message time_created
                    row.get::<_, Option<String>>(3)?.unwrap_or_default(), // message data
                    row.get::<_, Option<i64>>(5)?.unwrap_or_default(),    // part sequence
                    row.get::<_, Option<i64>>(6)?.unwrap_or_default(),    // part time_created
                    row.get::<_, Option<String>>(7)?.unwrap_or_default(), // part data
                    row.get::<_, Option<String>>(4)?.unwrap_or_default(), // part id
                ))
            })
            .with_context(|| format!("read messages in {}", path.display()))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        row_count += rows.len() as u64;
        for (
            _message_sequence,
            message_time_created,
            message_data,
            _part_sequence,
            part_time_created,
            part_data,
            part_id,
        ) in rows
        {
            let Ok(message) = serde_json::from_str::<Value>(&message_data) else {
                diagnostics.malformed_json_lines += 1;
                continue;
            };
            let Ok(part) = serde_json::from_str::<Value>(&part_data) else {
                diagnostics.malformed_json_lines += 1;
                continue;
            };
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("assistant");
            let part_type = part.get("type").and_then(Value::as_str).unwrap_or("");
            let ts = [
                part_time_created,
                message_time_created,
                session.time_created,
            ]
            .iter()
            .find(|&&candidate| candidate > 0)
            .copied()
            .unwrap_or_default()
            .max(0) as u64;
            let mut links = RecordLinks::default();
            if !part_id.is_empty() {
                links.event_id = Some(part_id.clone());
            }
            if let Some(parent) = session.parent_id.as_deref() {
                links.parent_session_id = Some(parent.to_string());
                links.conversation_kind = Some(ConversationKind::Subagent.as_str().to_string());
            } else {
                links.conversation_kind = Some(ConversationKind::Main.as_str().to_string());
            }
            let base = || Record {
                source: SourceKind::Zcode,
                doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                ts,
                project: project
                    .clone()
                    .unwrap_or_else(|| SourceKind::Zcode.label().to_string()),
                session_id: session.id.clone(),
                turn_id,
                role: String::new(),
                text: String::new(),
                tool_name: None,
                tool_input: None,
                tool_output: None,
                links: links.clone(),
                source_path: source_path.to_string(),
            };
            match part_type {
                "text" => {
                    let Some(text) = part.get("text").and_then(Value::as_str) else {
                        continue;
                    };
                    emit(Record {
                        role: role.to_string(),
                        text: text.to_string(),
                        ..base()
                    })?;
                    turn_id += 1;
                }
                "reasoning" => {
                    if !include_reasoning {
                        continue;
                    }
                    let Some(text) = part.get("text").and_then(Value::as_str) else {
                        continue;
                    };
                    emit(Record {
                        role: "reasoning".to_string(),
                        text: text.to_string(),
                        ..base()
                    })?;
                    turn_id += 1;
                }
                "tool" => {
                    let tool_name = part
                        .get("tool")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                        .to_string();
                    let input = part
                        .get("state")
                        .and_then(|state| state.get("input"))
                        .and_then(value_to_string);
                    let is_error =
                        part.pointer("/state/status").and_then(Value::as_str) == Some("error");
                    let output = part
                        .get("state")
                        .and_then(|state| state.get("output"))
                        .and_then(value_to_string)
                        .or_else(|| part.pointer("/state/error").and_then(value_to_string))
                        .or_else(|| is_error.then(|| "[tool error]".to_string()));
                    let call_id = part
                        .get("callID")
                        .and_then(Value::as_str)
                        .filter(|id| !id.is_empty())
                        .unwrap_or(&part_id)
                        .to_string();
                    let mut tool_links = links.clone();
                    tool_links.event_id = Some(call_id.clone());
                    emit(Record {
                        role: "tool_use".to_string(),
                        text: input.clone().unwrap_or_default(),
                        tool_name: Some(tool_name.clone()),
                        tool_input: input,
                        links: tool_links,
                        ..base()
                    })?;
                    if let Some(output) = output {
                        let mut result_links = links.clone();
                        result_links.event_id = Some(format!("{part_id}:result"));
                        result_links.parent_event_id = Some(call_id.clone());
                        result_links.parent_tool_use_id = Some(call_id);
                        result_links.tool_result_is_error = Some(is_error);
                        emit(Record {
                            role: "tool_result".to_string(),
                            text: output.clone(),
                            tool_name: Some(tool_name),
                            tool_output: Some(output),
                            links: result_links,
                            ..base()
                        })?;
                    }
                    turn_id += 1;
                }
                "step-start" | "step-finish" | "timeline" => {}
                other => {
                    diagnostics.increment_unknown_top_level(&format!("part_type_{other}"));
                }
            }
        }
    }
    drop(messages_statement);
    drop(sessions_statement);
    transaction.commit()?;
    Ok(IndexParseOutput {
        offset: row_count,
        turn_id,
        legacy_turn_id: None,
        pending_tool_calls: state.pending_tool_calls,
        session_id: selected_session.map(str::to_owned),
        diagnostics,
        session_cwd,
    })
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        other => serde_json::to_string(other).ok(),
    }
}

pub(crate) fn parse_usage_file(path: &Path) -> Result<UsageParseOutput> {
    let mut wal = path.as_os_str().to_os_string();
    wal.push("-wal");
    let wal = PathBuf::from(wal);
    let wal_before = UsageDependency::from_path_or_absent(&wal);
    let conn = open_readonly(path)?;
    let transaction = conn.unchecked_transaction()?;
    let tables = table_names(&transaction)?;
    let source_path: Arc<str> = Arc::from(path.to_string_lossy().to_string());
    let mut events = Vec::new();
    if tables.contains("model_usage") {
        let directories: HashMap<String, Option<String>> = if tables.contains("session") {
            let mut statement = transaction
                .prepare("SELECT id, directory FROM session")
                .with_context(|| format!("query sessions in {}", path.display()))?;
            statement
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
                })?
                .filter_map(Result::ok)
                .collect()
        } else {
            HashMap::new()
        };
        let mut statement = transaction
            .prepare(
                "SELECT id, session_id, logical_request_id, assistant_message_id, \
                 provider_id, model_id, started_at, input_tokens, output_tokens, \
                 reasoning_tokens, cache_creation_input_tokens, cache_read_input_tokens \
                 FROM model_usage ORDER BY started_at, id",
            )
            .with_context(|| format!("query model usage in {}", path.display()))?;
        let rows = statement
            .query_map([], |row| {
                Ok(ModelUsageRow {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    logical_request_id: row.get(2)?,
                    assistant_message_id: row.get(3)?,
                    provider_id: row.get(4)?,
                    model_id: row.get(5)?,
                    started_at: row.get(6)?,
                    input_tokens: row.get(7)?,
                    output_tokens: row.get(8)?,
                    reasoning_tokens: row.get(9)?,
                    cache_creation_input_tokens: row.get(10)?,
                    cache_read_input_tokens: row.get(11)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for (order, row) in rows.into_iter().enumerate() {
            let mut tokens = TokenBuckets::disjoint(
                row.input_tokens.max(0) as u64,
                row.cache_read_input_tokens.max(0) as u64,
                row.cache_creation_input_tokens.max(0) as u64,
                row.output_tokens.max(0) as u64,
            );
            // Reasoning is a subset of the reported output, not an extra bucket.
            tokens.reasoning = (row.reasoning_tokens.max(0) as u64).min(tokens.output);
            if tokens.additive_total() == 0 {
                continue;
            }
            let project = row
                .session_id
                .as_deref()
                .and_then(|id| directories.get(id))
                .cloned()
                .flatten();
            let sidechain = row
                .session_id
                .as_deref()
                .is_some_and(|id| id.starts_with("sess_subagent"));
            events.push(UsageEvent {
                source: "zcode",
                source_path: source_path.clone(),
                source_record_id: Some(row.id),
                session_id: row.session_id.filter(|value| !value.is_empty()),
                request_id: row.logical_request_id.filter(|value| !value.is_empty()),
                message_id: row.assistant_message_id.filter(|value| !value.is_empty()),
                timestamp_ms: row.started_at.max(0) as u64,
                project,
                provider: row.provider_id.filter(|value| !value.is_empty()),
                model: row.model_id.filter(|value| !value.is_empty()),
                tokens,
                credits: None,
                token_usage_available: true,
                // ZCode reports no per-request cost (subscription billing).
                source_cost_usd: None,
                cost_authoritative: false,
                dedupe_confidence: "exact",
                conservative_undercount: false,
                cache_chain_excluded: false,
                permission_review: false,
                sidechain,
                source_order: order as u64,
            });
        }
    }
    // Keep sessions and model usage on the same SQLite snapshot, and never
    // cache a read that raced a WAL commit.
    transaction.commit()?;
    let wal_after = UsageDependency::from_path_or_absent(&wal);
    Ok(UsageParseOutput {
        events,
        cacheable: wal_before == wal_after,
        deps: vec![wal_after],
    })
}

struct ModelUsageRow {
    id: String,
    session_id: Option<String>,
    logical_request_id: Option<String>,
    assistant_message_id: Option<String>,
    provider_id: Option<String>,
    model_id: Option<String>,
    started_at: i64,
    input_tokens: i64,
    output_tokens: i64,
    reasoning_tokens: i64,
    cache_creation_input_tokens: i64,
    cache_read_input_tokens: i64,
}

/// Working directory of one session, for analytics and session metadata.
pub fn session_cwd(path: &Path, session_id: &str) -> Option<PathBuf> {
    let virtual_session = split_virtual_path(path);
    let path = virtual_session
        .as_ref()
        .map_or(path, |(database, _)| database.as_path());
    let conn = open_readonly(path).ok()?;
    conn.query_row(
        "SELECT directory FROM session WHERE id = ?1",
        [session_id],
        |row| row.get::<_, Option<String>>(0),
    )
    .ok()
    .flatten()
    .filter(|directory| !directory.is_empty())
    .map(PathBuf::from)
}

/// Stored title of one session (ZCode titles every session), for labels.
pub fn session_title(path: &Path, session_id: &str) -> Option<String> {
    let virtual_session = split_virtual_path(path);
    let path = virtual_session
        .as_ref()
        .map_or(path, |(database, _)| database.as_path());
    let conn = open_readonly(path).ok()?;
    conn.query_row(
        "SELECT title FROM session WHERE id = ?1",
        [session_id],
        |row| row.get::<_, Option<String>>(0),
    )
    .ok()
    .flatten()
    .filter(|title| !title.trim().is_empty())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn fixture_db(path: &Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE session (id TEXT PRIMARY KEY, parent_id TEXT, directory TEXT, time_created INTEGER);
             CREATE TABLE message (id TEXT, session_id TEXT, sequence INTEGER, time_created INTEGER, data TEXT);
             CREATE TABLE part (id TEXT, message_id TEXT, session_id TEXT, sequence INTEGER, time_created INTEGER, data TEXT);
             CREATE TABLE model_usage (id TEXT PRIMARY KEY, session_id TEXT, logical_request_id TEXT, assistant_message_id TEXT, provider_id TEXT, model_id TEXT, started_at INTEGER, input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER, cache_creation_input_tokens INTEGER, cache_read_input_tokens INTEGER);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session VALUES ('sess_main', NULL, '/work/nipponhomes', 1000)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session VALUES ('sess_subagent_child', 'sess_main', '/work/nipponhomes', 1100)",
            [],
        )
        .unwrap();
        let messages: &[(&str, &str, i64, i64, &str)] = &[
            (
                "msg_u",
                "sess_main",
                0,
                1000,
                r#"{"role":"user","time":{"created":1000}}"#,
            ),
            (
                "msg_a",
                "sess_main",
                1,
                2000,
                r#"{"role":"assistant","modelID":"GLM-5.3","time":{"created":2000}}"#,
            ),
            (
                "msg_s",
                "sess_subagent_child",
                0,
                3000,
                r#"{"role":"user","time":{"created":3000}}"#,
            ),
        ];
        for (id, session, sequence, time, data) in messages {
            conn.execute(
                "INSERT INTO message VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![id, session, sequence, time, data],
            )
            .unwrap();
        }
        let parts: &[(&str, &str, i64, i64, &str)] = &[
            (
                "p_u",
                "msg_u",
                0,
                1000,
                r#"{"type":"text","text":"audit the crawler"}"#,
            ),
            (
                "p_t",
                "msg_a",
                0,
                2100,
                r#"{"type":"tool","callID":"call_1","tool":"Bash","state":{"status":"completed","input":{"command":"ls"},"output":"total 0"}}"#,
            ),
            (
                "p_r",
                "msg_a",
                1,
                2200,
                r#"{"type":"reasoning","text":"thinking"}"#,
            ),
            (
                "p_a",
                "msg_a",
                2,
                2300,
                r#"{"type":"text","text":"all clear"}"#,
            ),
            (
                "p_step",
                "msg_a",
                3,
                2350,
                r#"{"type":"step-finish","reason":"tool-calls"}"#,
            ),
            ("p_new", "msg_a", 4, 2400, r#"{"type":"future_marker"}"#),
            (
                "p_s",
                "msg_s",
                0,
                3000,
                r#"{"type":"text","text":"subagent prompt"}"#,
            ),
        ];
        for (id, message, sequence, time, data) in parts {
            conn.execute(
                "INSERT INTO part VALUES (?1, ?2, 'sess_main', ?3, ?4, ?5)",
                rusqlite::params![id, message, sequence, time, data],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO model_usage VALUES ('usage_1', 'sess_main', 'req_1', NULL, 'builtin:zai', 'GLM-5.3', 2000, 100, 40, 10, 5, 700), \
             ('usage_2', 'sess_subagent_child', 'req_2', 'msg_s', 'builtin:zai', 'GLM-5.3-Flash', 3000, 50, 0, 0, 0, 0), \
             ('usage_0', 'sess_main', 'req_0', NULL, 'builtin:zai', 'GLM-5.3', 1000, 0, 0, 0, 0, 0)",
            [],
        )
        .unwrap();
    }

    #[test]
    fn discovery_honors_explicit_roots_replacing_the_default() {
        let temp = tempfile::tempdir().unwrap();
        let explicit = temp.path().join("explicit");
        let default_home = temp.path().join("home");
        let synced = temp.path().join("synced");
        for root in [&explicit, &synced] {
            let db = root.join("cli").join("db").join("db.sqlite");
            std::fs::create_dir_all(db.parent().unwrap()).unwrap();
            std::fs::write(&db, b"x").unwrap();
        }
        let db = default_home
            .join(".zcode")
            .join("cli")
            .join("db")
            .join("db.sqlite");
        std::fs::create_dir_all(db.parent().unwrap()).unwrap();
        std::fs::write(&db, b"x").unwrap();

        let explicit_roots = format!("{},{}", explicit.display(), synced.display());
        let roots = roots_for(Some(std::ffi::OsStr::new(&explicit_roots)), &default_home);
        let files = discover_from_roots(&roots);
        let paths: Vec<String> = files
            .iter()
            .map(|file| file.path.to_string_lossy().into_owned())
            .collect();
        assert_eq!(paths.len(), 2);
        assert!(paths.iter().all(|path| !path.contains(".zcode")));
        assert!(files.iter().all(|file| file.source == SourceKind::Zcode));

        let default = roots_for(None, &default_home);
        assert_eq!(default, vec![default_home.join(".zcode")]);
    }

    #[test]
    fn records_carry_roles_tools_reasoning_and_hierarchy() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("db.sqlite");
        fixture_db(&path);

        let mut records = Vec::new();
        parse_index_records(
            &path,
            IndexParseState::default(),
            true,
            &AtomicU64::new(0),
            |record| {
                records.push(record);
                Ok(())
            },
        )
        .unwrap();

        let user = records
            .iter()
            .find(|r| r.text == "audit the crawler")
            .unwrap();
        assert_eq!(user.role, "user");
        assert_eq!(user.project, "nipponhomes");
        assert_eq!(user.session_id, "sess_main");
        assert_eq!(user.ts, 1000);
        assert_eq!(user.links.conversation_kind.as_deref(), Some("main"));

        let tool = records.iter().find(|r| r.tool_name.is_some()).unwrap();
        assert_eq!(tool.role, "tool_use");
        assert_eq!(tool.tool_name.as_deref(), Some("Bash"));
        assert_eq!(tool.tool_input.as_deref(), Some(r#"{"command":"ls"}"#));
        assert_eq!(tool.tool_output, None);
        assert_eq!(tool.ts, 2100);
        let result = records.iter().find(|r| r.role == "tool_result").unwrap();
        assert_eq!(result.text, "total 0");
        assert_eq!(result.tool_output.as_deref(), Some("total 0"));
        assert_eq!(result.links.parent_tool_use_id, tool.links.event_id);
        assert_ne!(result.links.event_id, tool.links.event_id);

        let reasoning = records.iter().find(|r| r.role == "reasoning").unwrap();
        assert_eq!(reasoning.text, "thinking");

        let subagent = records
            .iter()
            .find(|r| r.text == "subagent prompt")
            .unwrap();
        assert_eq!(subagent.session_id, "sess_subagent_child");
        assert_eq!(
            subagent.links.parent_session_id.as_deref(),
            Some("sess_main")
        );
        assert_eq!(
            subagent.links.conversation_kind.as_deref(),
            Some("subagent")
        );

        assert_eq!(records.len(), 6);
    }

    #[test]
    fn reasoning_is_gated_by_include_reasoning() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("db.sqlite");
        fixture_db(&path);
        let mut records = Vec::new();
        parse_index_records(
            &path,
            IndexParseState::default(),
            false,
            &AtomicU64::new(0),
            |record| {
                records.push(record);
                Ok(())
            },
        )
        .unwrap();
        assert!(!records.iter().any(|r| r.role == "reasoning"));
        assert_eq!(records.len(), 5);
    }

    #[test]
    fn usage_events_map_model_request_rows() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("db.sqlite");
        fixture_db(&path);
        let output = parse_usage_file(&path).unwrap();
        assert_eq!(output.events.len(), 2);

        let mut events = output.events;
        events.sort_by_key(|event| event.timestamp_ms);
        let main = events.first().unwrap();
        assert_eq!(main.session_id.as_deref(), Some("sess_main"));
        assert_eq!(main.model.as_deref(), Some("GLM-5.3"));
        assert_eq!(main.tokens.uncached_input, 100);
        assert_eq!(main.tokens.cache_read, 700);
        assert_eq!(main.tokens.cache_write, 5);
        assert_eq!(main.tokens.output, 40);
        assert_eq!(main.tokens.reasoning, 10);
        assert_eq!(main.project.as_deref(), Some("/work/nipponhomes"));
        assert!(!main.sidechain);

        let subagent = events.last().unwrap();
        assert!(subagent.sidechain);
        assert_eq!(subagent.tokens.output, 0);
        assert_eq!(subagent.tokens.uncached_input, 50);
        assert_eq!(output.deps.len(), 1);
    }

    #[test]
    fn session_metadata_helpers_read_single_rows() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("db.sqlite");
        fixture_db(&path);
        // The fixture has no title column; titles fall back to none.
        assert_eq!(
            session_cwd(&path, "sess_main"),
            Some(PathBuf::from("/work/nipponhomes"))
        );
        assert_eq!(session_cwd(&path, "missing"), None);
    }

    #[test]
    fn virtual_sessions_round_trip_and_parse_only_the_selected_session() {
        let temp = tempfile::tempdir().unwrap();
        let database = temp.path().join("db.sqlite");
        fixture_db(&database);
        for id in ["sess_main", "../unicode/東京\\session", ""] {
            let path = virtual_path(&database, id);
            assert_eq!(path.parent(), Some(database.as_path()));
            assert_eq!(
                split_virtual_path(&path),
                Some((database.clone(), id.to_string()))
            );
            assert!(matches_path(path.to_str().unwrap()));
        }
        assert!(split_virtual_path(&database.join("zcode-session-zz")).is_none());
        let path = virtual_path(&database, "sess_main");
        let mut records = Vec::new();
        let parsed = parse_index_records(
            &path,
            IndexParseState::default(),
            true,
            &AtomicU64::new(0),
            |record| {
                records.push(record);
                Ok(())
            },
        )
        .unwrap();
        let sessions = enumerate_sessions(&database).unwrap();
        let session = sessions
            .iter()
            .find(|session| session.id == "sess_main")
            .unwrap();
        assert_eq!(parsed.offset, session.size);
        assert_eq!(parsed.session_id.as_deref(), Some("sess_main"));
        assert!(records.iter().all(|record| record.session_id == "sess_main"
            && record.source_path == path.to_string_lossy()));
        assert_eq!(
            session_cwd(&path, "sess_main"),
            Some(PathBuf::from("/work/nipponhomes"))
        );
    }

    #[test]
    fn fingerprints_detect_content_edits_and_ignore_usage_writes() {
        let temp = tempfile::tempdir().unwrap();
        let database = temp.path().join("db.sqlite");
        fixture_db(&database);
        let before = enumerate_sessions(&database).unwrap();
        let conn = Connection::open(&database).unwrap();
        conn.execute("UPDATE model_usage SET input_tokens = input_tokens + 1", [])
            .unwrap();
        let usage_only = enumerate_sessions(&database).unwrap();
        assert!(
            before
                .iter()
                .zip(&usage_only)
                .all(|(a, b)| a.fingerprint == b.fingerprint)
        );
        conn.execute(
            "UPDATE part SET data = '{\"type\":\"text\",\"text\":\"changed\"}' WHERE id = 'p_u'",
            [],
        )
        .unwrap();
        let changed = enumerate_sessions(&database).unwrap();
        assert_ne!(before[0].fingerprint, changed[0].fingerprint);
        assert_eq!(before[1].fingerprint, changed[1].fingerprint);
        assert_eq!(before[0].size, changed[0].size);
        conn.execute(
            "UPDATE session SET directory = '/new/project' WHERE id = 'sess_main'",
            [],
        )
        .unwrap();
        let metadata = enumerate_sessions(&database).unwrap();
        assert_ne!(changed[0].fingerprint, metadata[0].fingerprint);
        assert_eq!(changed[1].fingerprint, metadata[1].fingerprint);
    }

    #[test]
    fn tool_outputs_and_errors_are_searchable_in_the_real_index() {
        use crate::index::{QueryOptions, SearchIndex};
        let temp = tempfile::tempdir().unwrap();
        let database = temp.path().join("db.sqlite");
        fixture_db(&database);
        let index = SearchIndex::open_or_create(&temp.path().join("index")).unwrap();
        let mut writer = index.writer().unwrap();
        let conn = Connection::open(&database).unwrap();
        let ids = AtomicU64::new(0);
        for (status, field, sentinel) in [
            ("completed", "output", "outputonlysentinel"),
            ("error", "error", "erroronlysentinel"),
        ] {
            let data = serde_json::json!({
                "type": "tool",
                "callID": "call_1",
                "tool": "Bash",
                "state": {"status": status, "input": {"command": "ls"}, (field): sentinel}
            });
            conn.execute(
                "UPDATE part SET data = ?1 WHERE id = 'p_t'",
                [data.to_string()],
            )
            .unwrap();
            parse_index_records(
                &virtual_path(&database, "sess_main"),
                IndexParseState::default(),
                false,
                &ids,
                |record| index.add_record(&mut writer, &record),
            )
            .unwrap();
        }
        writer.commit().unwrap();
        for (sentinel, is_error) in [("outputonlysentinel", false), ("erroronlysentinel", true)] {
            let matches = index
                .search(&QueryOptions {
                    query: sentinel.to_string(),
                    project: None,
                    role: None,
                    tool: None,
                    session_id: None,
                    session_scope: None,
                    source: None,
                    since: None,
                    until: None,
                    limit: 10,
                })
                .unwrap();
            assert_eq!(matches.len(), 1);
            let result = &matches[0].1;
            assert_eq!(result.role, "tool_result");
            assert_eq!(result.text, sentinel);
            assert_eq!(result.links.parent_tool_use_id.as_deref(), Some("call_1"));
            assert_eq!(result.links.tool_result_is_error, Some(is_error));
        }
    }

    #[test]
    fn usage_repository_filter_resolves_a_differently_named_worktree() {
        use crate::analytics::ProjectGrouping;
        use crate::test_support::{EnvVarGuard, env_lock};
        use crate::types::SourceFilter;
        use crate::usage::{UsageQuery, scan_usage};
        let _guard = env_lock();
        let temp = tempfile::tempdir().unwrap();
        let repository = temp.path().join("canonical-repository");
        let worktree = temp.path().join("feature-checkout");
        std::fs::create_dir_all(&repository).unwrap();
        for args in [
            vec!["init"],
            vec![
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--allow-empty",
                "-m",
                "Initial",
            ],
            vec![
                "worktree",
                "add",
                "-b",
                "feature",
                worktree.to_str().unwrap(),
            ],
        ] {
            let output = std::process::Command::new("git")
                .args(args)
                .current_dir(&repository)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let root = temp.path().join("zcode");
        let database = root.join("cli/db/db.sqlite");
        std::fs::create_dir_all(database.parent().unwrap()).unwrap();
        fixture_db(&database);
        Connection::open(&database)
            .unwrap()
            .execute(
                "UPDATE session SET directory = ?1",
                [worktree.to_str().unwrap()],
            )
            .unwrap();
        let _env = EnvVarGuard::set_os(&[("ZCODE_HOME", Some(root.as_os_str()))]);
        let mut query = UsageQuery {
            source: Some(SourceFilter::Zcode),
            project: Some("canonical-repository".to_string()),
            project_grouping: ProjectGrouping::Repository,
            include_events: true,
            ..UsageQuery::default()
        };
        let repository_usage = scan_usage(&query).unwrap();
        assert_eq!(repository_usage.events, 2);
        query.project_grouping = ProjectGrouping::Flat;
        assert_eq!(scan_usage(&query).unwrap().events, 0);
        query.project = Some("feature-checkout".to_string());
        assert_eq!(scan_usage(&query).unwrap().events, 2);
    }

    #[test]
    fn matches_zcode_store_paths() {
        assert!(matches_path("/root/.zcode/cli/db/db.sqlite"));
        assert!(matches_path("C:\\Users\\u\\.zcode\\cli\\db\\db.sqlite"));
        assert!(!matches_path("/root/.zcode/v2/tasks-index.sqlite"));
        assert!(!matches_path("/root/.zcode/cli/db/db.sqlite-wal"));
    }
}
