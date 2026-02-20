use crate::tools::Tool;
use minijinja::{context, Environment};
#[cfg(feature = "python")]
use pyo3::pyclass;
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
                escaped.content = self.escape_text(&escaped.content);
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
}
