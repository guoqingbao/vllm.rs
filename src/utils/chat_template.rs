use crate::tools::Tool;
use minijinja::{context, Environment};
use once_cell::sync::Lazy;
#[cfg(feature = "python")]
use pyo3::pyclass;
use regex::Regex;
use tokenizers::Tokenizer;

#[cfg(feature = "python")]
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[pyclass]
pub struct Message {
    #[pyo3(get)]
    pub role: String,
    #[pyo3(get)]
    pub content: String,
    pub num_images: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[cfg(not(feature = "python"))]
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub num_images: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[cfg(not(feature = "python"))]
impl Message {
    pub fn new(role: String, content: String, num_images: usize) -> Self {
        Message {
            role,
            content,
            num_images,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ApplyChatTemplateError {
    #[error("failed to add chat template")]
    AddTemplateError(#[source] minijinja::Error),
    #[error("failed to get chat template")]
    GetTemplateError(#[source] minijinja::Error),
    #[error("failed to render chat template")]
    RenderTemplateError(#[source] minijinja::Error),
}

fn escape_special_tokens_in_text(
    content: &str,
    escape_tokens: &[String],
    preserve_tokens: &[String],
) -> String {
    if escape_tokens.is_empty() || content.is_empty() {
        return content.to_string();
    }

    // Protect model-required markers (e.g. multimodal image markers) from
    // escaping by swapping them to temporary sentinels first.
    let mut protected = content.to_string();
    let mut sentinels: Vec<(String, String)> = Vec::new();
    for (idx, token) in preserve_tokens.iter().enumerate() {
        if token.is_empty() || !protected.contains(token) {
            continue;
        }
        let sentinel = format!("__VLLM_RS_PRESERVE_TOKEN_{}__", idx);
        protected = protected.replace(token, &sentinel);
        sentinels.push((sentinel, token.clone()));
    }

    let mut escaped = protected;
    for token in escape_tokens {
        if token.is_empty() {
            continue;
        }
        // Insert ZWNJ after '<' so textual tags remain visible but cannot be
        // recognized as tokenizer special/added-token spans.
        let escaped_token = if let Some(rest) = token.strip_prefix('<') {
            format!("<\u{200C}{}", rest)
        } else {
            format!("{}\u{200C}", token)
        };
        escaped = escaped.replace(token, &escaped_token);
    }

    for (sentinel, token) in sentinels {
        escaped = escaped.replace(&sentinel, &token);
    }

    escaped
}

fn should_escape_marker(token: &str) -> bool {
    if token.is_empty() || token.len() < 3 {
        return false;
    }
    let Some(first) = token.chars().next() else {
        return false;
    };
    matches!(first, '<' | '[' | '{' | '(') || token.contains('|')
}

fn should_escape_nested_xml_tool_markers(tool_markers: &[&str]) -> bool {
    tool_markers
        .iter()
        .any(|marker| marker.starts_with('<') && marker.contains("tool_call"))
}

#[derive(Clone, Debug)]
pub struct ChatTemplate {
    system_message: Option<String>,
    chat_template: Option<String>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    messages: Vec<Message>,
    escape_tokens: Vec<String>,
    preserve_tokens: Vec<String>,
    add_generation_prompt: bool,
    enable_thinking: bool,
}

impl ChatTemplate {
    pub fn collect_escape_tokens(tokenizer: &Tokenizer, tool_markers: &[&str]) -> Vec<String> {
        let mut tokens = tokenizer
            .get_added_tokens_decoder()
            .into_values()
            .filter(|added| added.special)
            .map(|added| added.content)
            .collect::<Vec<_>>();

        for marker in tool_markers {
            if should_escape_marker(marker) {
                tokens.push((*marker).to_string());
            }
        }

        if should_escape_nested_xml_tool_markers(tool_markers) {
            tokens.extend(
                ["<function=", "</function>", "<parameter=", "</parameter>"]
                    .into_iter()
                    .map(|s| s.to_string()),
            );
        }

        // Escape longest markers first to avoid partial replacement ordering issues.
        tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));
        tokens.dedup();
        tokens
    }

    pub fn new(
        system_message: Option<String>,
        chat_template: Option<String>,
        bos_token: Option<String>,
        eos_token: Option<String>,
        prompt: Option<String>,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Self {
        let mut template = ChatTemplate {
            system_message: system_message.clone(),
            chat_template,
            bos_token,
            eos_token,
            messages: Vec::new(),
            escape_tokens: Vec::new(),
            preserve_tokens: Vec::new(),
            add_generation_prompt,
            enable_thinking,
        };
        if system_message.is_some() {
            template.append_message(
                "system".to_string(),
                template.system_message.clone().unwrap_or_default(),
                0,
            );
        }
        if let Some(prompt) = prompt {
            template.append_message("user".to_string(), prompt, 0);
        }
        template
    }

    pub fn append_message(&mut self, role: String, content: String, num_images: usize) {
        self.messages.push(Message {
            role,
            content,
            num_images,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    pub fn set_messages(&mut self, messages: &Vec<Message>) {
        self.messages.clear();
        self.messages.extend(messages.clone());
    }

    pub fn set_enable_thinking(&mut self, enable: bool) {
        self.enable_thinking = enable;
    }

    pub fn template_source(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    pub fn set_escape_tokens(&mut self, mut tokens: Vec<String>) {
        tokens.retain(|token| !token.is_empty());
        tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));
        tokens.dedup();
        self.escape_tokens = tokens;
    }

    pub fn set_preserve_tokens(&mut self, mut tokens: Vec<String>) {
        tokens.retain(|token| !token.is_empty());
        tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));
        tokens.dedup();
        self.preserve_tokens = tokens;
    }

    pub fn escape_text(&self, content: &str) -> String {
        escape_special_tokens_in_text(content, &self.escape_tokens, &self.preserve_tokens)
    }

    #[allow(dead_code)]
    fn clear_message(&mut self) {
        self.messages.clear()
    }

    fn escaped_messages_for_render(&self) -> Vec<Message> {
        if self.escape_tokens.is_empty() {
            return self.messages.clone();
        }
        self.messages
            .iter()
            .map(|message| {
                let mut escaped = message.clone();
                // System/developer prompts can include engine-defined structural
                // tool-call instructions that must remain exact (e.g. <tool_call>).
                // Escape only user/assistant/tool payloads.
                if !matches!(escaped.role.as_str(), "system" | "developer") {
                    escaped.content = self.escape_text(&escaped.content);
                }
                escaped
            })
            .collect()
    }

    pub fn apply_chat_template(
        &self,
        tools: &Vec<Tool>,
        log: bool,
    ) -> Result<String, ApplyChatTemplateError> {
        if self.chat_template.is_none() {
            return Err(ApplyChatTemplateError::GetTemplateError(
                minijinja::Error::new(minijinja::ErrorKind::CannotDeserialize, "Not found!"),
            ));
        }
        let mut env = Environment::new();
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        let template = self.chat_template.as_ref().unwrap();
        let mut template = template.replace("[::-1]", "|reverse");
        if template.find("{{ meta }}").is_some() {
            template = template.replace("{%- set meta = message.get(\"metadata\", \"\") %}", "");
            template = template.replace("{{ meta }}", "");
        }
        env.add_template("vllm.rs", template.as_str())
            .map_err(ApplyChatTemplateError::AddTemplateError)?;
        let template = env
            .get_template("vllm.rs")
            .map_err(ApplyChatTemplateError::GetTemplateError)?;

        let render_messages = self.escaped_messages_for_render();
        if log {
            tracing::info!("messages {:?}", render_messages);
        }
        template
            .render(context! {
              messages => render_messages,
              add_generation_prompt => self.add_generation_prompt,
              bos_token => self.bos_token,
              eos_token => self.eos_token,
              enable_thinking => self.enable_thinking,
              tools => tools,
            })
            .map_err(ApplyChatTemplateError::RenderTemplateError)
    }

    pub fn eos_token(&self) -> Option<&str> {
        self.eos_token.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Rendered-prompt repair for prefix-cache alignment
// ---------------------------------------------------------------------------

static GENERATION_PROMPT_MULTI_LITERAL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?s)\{%-?\s*if\s+add_generation_prompt\s*-?%\}(?P<body>.*?)\{%-?\s*endif\s*-?%\}"#,
    )
    .expect("valid generation prompt block regex")
});

static TEMPLATE_STRING_LITERAL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"\{\{\-?\s*['"](?P<lit>.*?)['"]\s*-?\}\}"#)
        .expect("valid template string literal regex")
});

static ENABLE_THINKING_FALSE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?s)enable_thinking\s+is\s+(defined\s+and\s+enable_thinking\s+is\s+)?false"#)
        .expect("valid enable_thinking false regex")
});

fn decode_template_string_literal(literal: &str) -> String {
    let mut decoded = String::with_capacity(literal.len());
    let mut chars = literal.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('n') => decoded.push('\n'),
                Some('r') => decoded.push('\r'),
                Some('t') => decoded.push('\t'),
                Some('\\') => decoded.push('\\'),
                Some('\'') => decoded.push('\''),
                Some('"') => decoded.push('"'),
                Some(other) => {
                    decoded.push('\\');
                    decoded.push(other);
                }
                None => decoded.push('\\'),
            }
        } else {
            decoded.push(ch);
        }
    }
    decoded
}

fn escaped_special_token(token: &str) -> String {
    if let Some(rest) = token.strip_prefix('<') {
        format!("<\u{200C}{}", rest)
    } else {
        format!("{}\u{200C}", token)
    }
}

/// Extracts the full generation-prompt literal that the template would emit
/// when `add_generation_prompt` is true.
///
/// Handles three patterns found across Qwen model families:
///   1. Single combined literal  (e.g. Qwen3-4B-Thinking)
///   2. Header literal + `enable_thinking` branch  (e.g. Qwen3.5)
///   3. Header literal only, no thinking branch  (e.g. Qwen3-Coder-Next, Qwen3-VL)
fn extract_generation_prompt_literal(chat_template: &str, enable_thinking: bool) -> Option<String> {
    let block_caps = GENERATION_PROMPT_MULTI_LITERAL_RE.captures(chat_template)?;
    let body = block_caps.name("body")?.as_str();

    let literals: Vec<String> = TEMPLATE_STRING_LITERAL_RE
        .captures_iter(body)
        .filter_map(|c| c.name("lit").map(|m| m.as_str().to_string()))
        .collect();

    if literals.is_empty() {
        return None;
    }

    let has_thinking_branch = body.contains("enable_thinking");
    if !has_thinking_branch {
        return Some(
            literals
                .iter()
                .map(|l| decode_template_string_literal(l))
                .collect::<String>(),
        );
    }

    // Template has an enable_thinking branch. Parse the structure:
    //   header literal(s)  →  before the `if enable_thinking` block
    //   disabled literal(s) → inside the `enable_thinking is false` branch
    //   enabled literal(s)  → inside the `else` branch
    let thinking_block_start = body.find("enable_thinking")?;

    let header_body = &body[..thinking_block_start];
    let header_literals: Vec<String> = TEMPLATE_STRING_LITERAL_RE
        .captures_iter(header_body)
        .filter_map(|c| c.name("lit").map(|m| m.as_str().to_string()))
        .collect();

    let thinking_body = &body[thinking_block_start..];

    let is_false_first = ENABLE_THINKING_FALSE_RE
        .is_match(&thinking_body[..thinking_body.find("else").unwrap_or(thinking_body.len())]);

    let branch_literals: Vec<Vec<String>> = thinking_body
        .split("{%- else")
        .chain(thinking_body.split("{% else"))
        .take(2)
        .map(|section| {
            TEMPLATE_STRING_LITERAL_RE
                .captures_iter(section)
                .filter_map(|c| c.name("lit").map(|m| m.as_str().to_string()))
                .collect()
        })
        .collect();

    let (disabled_lits, enabled_lits) = if branch_literals.len() >= 2 {
        if is_false_first {
            (&branch_literals[0], &branch_literals[1])
        } else {
            (&branch_literals[1], &branch_literals[0])
        }
    } else {
        return None;
    };

    let suffix_lits = if enable_thinking {
        enabled_lits
    } else {
        disabled_lits
    };

    let mut result = String::new();
    for lit in &header_literals {
        result.push_str(&decode_template_string_literal(lit));
    }
    for lit in suffix_lits {
        result.push_str(&decode_template_string_literal(lit));
    }
    Some(result)
}

/// Extracts the end-of-turn delimiter from the chat template by looking at how
/// assistant messages are terminated. Falls back to the configured eos_token.
fn extract_eot_delimiter(chat_template: &str, eos_token: Option<&str>) -> Option<String> {
    static EOT_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r#"(?s)message\.role\s*==\s*['"]assistant['"].*?\{\{\-?\s*['"](?P<eot><\|[^|]+\|>)['"]\s*-?\}\}"#)
            .expect("valid eot regex")
    });

    if let Some(caps) = EOT_RE.captures(chat_template) {
        if let Some(eot) = caps.name("eot") {
            let decoded = decode_template_string_literal(eot.as_str());
            if decoded.contains("end") || decoded.contains("eot") {
                return Some(decoded);
            }
        }
    }

    eos_token.map(|s| s.to_string())
}

/// Holds the extracted repair parameters for a specific template + thinking mode.
/// `None` fields mean no repair is needed for that aspect.
#[derive(Debug, Clone)]
pub struct RenderedPromptRepairer {
    assistant_header: String,
    eot_delimiter: String,
    start_marker: Option<String>,
    end_marker: Option<String>,
    scaffold: Option<String>,
}

impl RenderedPromptRepairer {
    /// Build a repairer from a chat template source and thinking mode.
    /// Returns `None` if no repair is possible (e.g. no assistant header found).
    pub fn from_template(
        chat_template: &str,
        eos_token: Option<&str>,
        enable_thinking: bool,
    ) -> Option<Self> {
        let generation_literal = extract_generation_prompt_literal(chat_template, enable_thinking)?;

        if generation_literal.is_empty() {
            return None;
        }

        let eot_delimiter = extract_eot_delimiter(chat_template, eos_token)
            .unwrap_or_else(|| "<|im_end|>".to_string());

        // Find the reasoning start marker within the generation literal.
        // We look for known reasoning markers that appear in the literal.
        let known_markers = [
            ("<think>", "</think>"),
            ("<thinking>", "</thinking>"),
            ("<reasoning>", "</reasoning>"),
            ("<reflection>", "</reflection>"),
            ("<internal>", "</internal>"),
        ];

        let mut found_start: Option<(usize, &str, &str)> = None;
        for (start, end) in &known_markers {
            if let Some(idx) = generation_literal.find(start) {
                found_start = Some((idx, start, end));
                break;
            }
        }

        let (assistant_header, start_marker, end_marker, scaffold) =
            if let Some((idx, start, end)) = found_start {
                let header = generation_literal[..idx].to_string();
                let suffix = generation_literal[idx..].to_string();
                (
                    header,
                    Some(start.to_string()),
                    Some(end.to_string()),
                    Some(suffix),
                )
            } else if generation_literal.contains("assistant") {
                (generation_literal, None, None, None)
            } else {
                return None;
            };

        if assistant_header.is_empty() {
            return None;
        }

        Some(Self {
            assistant_header,
            eot_delimiter,
            start_marker,
            end_marker,
            scaffold,
        })
    }

    /// Build a repairer from a `ChatTemplate` instance.
    pub fn from_chat_template(template: &ChatTemplate, enable_thinking: bool) -> Option<Self> {
        let source = template.template_source()?;
        Self::from_template(source, template.eos_token(), enable_thinking)
    }

    /// Returns true if this repairer has a reasoning scaffold to insert.
    pub fn has_reasoning_scaffold(&self) -> bool {
        self.scaffold.is_some()
    }

    /// Apply the repair to a rendered prompt. Returns `None` if no changes needed.
    pub fn repair(&self, base_prompt: &str) -> Option<String> {
        let (Some(start_marker), Some(end_marker), Some(scaffold)) =
            (&self.start_marker, &self.end_marker, &self.scaffold)
        else {
            return None;
        };

        let escaped_start = escaped_special_token(start_marker);
        let escaped_end = escaped_special_token(end_marker);

        // The "opening scaffold" is the part before the end marker (if the
        // scaffold contains a paired close, e.g. `<think>\n\n</think>\n\n`).
        let opening_scaffold = if let Some(idx) = scaffold.find(end_marker.as_str()) {
            &scaffold[..idx]
        } else {
            scaffold.as_str()
        };

        let mut cursor = 0usize;
        let mut repaired = String::with_capacity(base_prompt.len() + 128);
        let mut changed = false;

        while let Some(rel_idx) = base_prompt[cursor..].find(&self.assistant_header) {
            let header_idx = cursor + rel_idx;
            let after_header = header_idx + self.assistant_header.len();
            repaired.push_str(&base_prompt[cursor..after_header]);

            let rest = &base_prompt[after_header..];
            let block_end = rest.find(self.eot_delimiter.as_str()).unwrap_or(rest.len());
            let block = &rest[..block_end];
            let trimmed = block.trim_start();

            let has_start = trimmed.starts_with(start_marker.as_str());
            let has_escaped_start = trimmed.starts_with(&escaped_start);
            let raw_end_present = block.contains(end_marker.as_str());
            let escaped_end_idx = block.find(&escaped_end);
            let has_end = raw_end_present || escaped_end_idx.is_some();

            let needs_prefix = !has_start;
            let needs_end_restore = escaped_end_idx.is_some();

            if needs_prefix || needs_end_restore {
                let prefix = if !needs_prefix || has_escaped_start {
                    ""
                } else if has_end {
                    opening_scaffold
                } else {
                    scaffold.as_str()
                };
                repaired.push_str(prefix);

                let mut content = block.to_string();
                if needs_prefix && has_escaped_start {
                    let ws_len = content.len() - content.trim_start().len();
                    content.replace_range(ws_len..ws_len + escaped_start.len(), start_marker);
                }
                if let Some(eidx) = content.find(&escaped_end) {
                    content.replace_range(eidx..eidx + escaped_end.len(), end_marker);
                }
                repaired.push_str(&content);
                changed = true;
            } else {
                repaired.push_str(block);
            }

            cursor = after_header + block_end;
        }

        if !changed {
            return None;
        }

        repaired.push_str(&base_prompt[cursor..]);
        Some(repaired)
    }

    /// Convenience: try to build a repairer and apply it in one step.
    /// Returns the repaired prompt or `None` if no repair was needed/possible.
    pub fn try_repair(
        chat_template: &str,
        eos_token: Option<&str>,
        enable_thinking: bool,
        base_prompt: &str,
    ) -> Option<String> {
        let repairer = Self::from_template(chat_template, eos_token, enable_thinking)?;
        repairer.repair(base_prompt)
    }
}

#[cfg(test)]
mod repair_tests {
    use super::*;

    const QWEN35_TEMPLATE: &str = r#"
{%- for message in messages %}
    {%- if message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
"#;

    const QWEN3_4B_TEMPLATE: &str = r#"
{%- for message in messages %}
    {%- if message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}
"#;

    const QWEN3_CODER_TEMPLATE: &str = r#"
{%- for message in messages %}
    {%- if message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"#;

    #[test]
    fn qwen35_thinking_enabled_extracts_correct_scaffold() {
        let r = RenderedPromptRepairer::from_template(QWEN35_TEMPLATE, Some("<|im_end|>"), true)
            .unwrap();
        assert_eq!(r.assistant_header, "<|im_start|>assistant\n");
        assert_eq!(r.start_marker.as_deref(), Some("<think>"));
        assert_eq!(r.end_marker.as_deref(), Some("</think>"));
        assert_eq!(r.scaffold.as_deref(), Some("<think>\n"));
    }

    #[test]
    fn qwen35_thinking_disabled_extracts_paired_scaffold() {
        let r = RenderedPromptRepairer::from_template(QWEN35_TEMPLATE, Some("<|im_end|>"), false)
            .unwrap();
        assert_eq!(r.assistant_header, "<|im_start|>assistant\n");
        assert_eq!(r.scaffold.as_deref(), Some("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn qwen3_4b_combined_literal_extracts_scaffold() {
        let r = RenderedPromptRepairer::from_template(QWEN3_4B_TEMPLATE, Some("<|im_end|>"), true)
            .unwrap();
        assert_eq!(r.assistant_header, "<|im_start|>assistant\n");
        assert_eq!(r.scaffold.as_deref(), Some("<think>\n"));
    }

    #[test]
    fn qwen3_coder_no_thinking_returns_no_scaffold() {
        let r =
            RenderedPromptRepairer::from_template(QWEN3_CODER_TEMPLATE, Some("<|im_end|>"), true)
                .unwrap();
        assert_eq!(r.assistant_header, "<|im_start|>assistant\n");
        assert!(!r.has_reasoning_scaffold());
    }

    #[test]
    fn no_repair_needed_when_template_has_no_thinking() {
        let prompt = "<|im_start|>assistant\nHello world<|im_end|>\n";
        let result = RenderedPromptRepairer::try_repair(
            QWEN3_CODER_TEMPLATE,
            Some("<|im_end|>"),
            true,
            prompt,
        );
        assert!(result.is_none());
    }

    #[test]
    fn repair_inserts_missing_think_prefix_enabled() {
        let prompt = "<|im_start|>user\nhi<|im_end|>\n\
                       <|im_start|>assistant\nThinking...\n</think>\nhello<|im_end|>\n";
        let repaired =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), true, prompt)
                .unwrap();
        assert!(repaired.contains("<|im_start|>assistant\n<think>\nThinking..."));
    }

    #[test]
    fn repair_inserts_full_paired_scaffold_when_disabled_and_no_end_marker() {
        let prompt = "<|im_start|>assistant\nVisible answer<|im_end|>\n";
        let repaired =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), false, prompt)
                .unwrap();
        assert!(
            repaired.starts_with("<|im_start|>assistant\n<think>\n\n</think>\n\nVisible answer")
        );
    }

    #[test]
    fn repair_inserts_opening_scaffold_when_end_marker_present() {
        let prompt = "<|im_start|>assistant\nThinking...\n</think>\nhello<|im_end|>\n";
        let repaired =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), false, prompt)
                .unwrap();
        assert!(repaired.starts_with("<|im_start|>assistant\n<think>\n\nThinking..."));
    }

    #[test]
    fn repair_restores_escaped_end_marker() {
        let prompt =
            "<|im_start|>assistant\n<think>\nreasoning\n<\u{200C}/think>\nanswer<|im_end|>\n";
        let repaired =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), true, prompt)
                .unwrap();
        assert_eq!(
            repaired,
            "<|im_start|>assistant\n<think>\nreasoning\n</think>\nanswer<|im_end|>\n"
        );
    }

    #[test]
    fn repair_with_qwen3_4b_combined_template() {
        let prompt = "<|im_start|>assistant\nSome reasoning\n</think>\nhello<|im_end|>\n";
        let repaired =
            RenderedPromptRepairer::try_repair(QWEN3_4B_TEMPLATE, Some("<|im_end|>"), true, prompt)
                .unwrap();
        assert!(repaired.contains("<|im_start|>assistant\n<think>\nSome reasoning"));
    }

    #[test]
    fn repair_no_change_when_prefix_already_present() {
        let prompt = "<|im_start|>assistant\n<think>\nreasoning\n</think>\nhello<|im_end|>\n";
        let result =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), true, prompt);
        assert!(result.is_none());
    }

    #[test]
    fn extract_generation_literal_qwen35_enabled() {
        let literal = extract_generation_prompt_literal(QWEN35_TEMPLATE, true).unwrap();
        assert_eq!(literal, "<|im_start|>assistant\n<think>\n");
    }

    #[test]
    fn extract_generation_literal_qwen35_disabled() {
        let literal = extract_generation_prompt_literal(QWEN35_TEMPLATE, false).unwrap();
        assert_eq!(literal, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }

    #[test]
    fn extract_generation_literal_qwen3_4b() {
        let literal = extract_generation_prompt_literal(QWEN3_4B_TEMPLATE, true).unwrap();
        assert_eq!(literal, "<|im_start|>assistant\n<think>\n");
    }

    #[test]
    fn extract_generation_literal_qwen3_coder() {
        let literal = extract_generation_prompt_literal(QWEN3_CODER_TEMPLATE, true).unwrap();
        assert_eq!(literal, "<|im_start|>assistant\n");
    }

    #[test]
    fn eot_delimiter_extracted_from_template() {
        let eot = extract_eot_delimiter(QWEN35_TEMPLATE, Some("<|im_end|>"));
        assert_eq!(eot.as_deref(), Some("<|im_end|>"));
    }

    #[test]
    fn eot_delimiter_falls_back_to_eos_token() {
        let eot = extract_eot_delimiter("no matching pattern here", Some("<|endoftext|>"));
        assert_eq!(eot.as_deref(), Some("<|endoftext|>"));
    }
}
