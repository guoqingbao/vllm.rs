use tokenizers::tokenizer::{Tokenizer, AddedToken};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum Category {
    Eos,
    Pad,
    Bos,
    Sep,
    Cls,
    Mask,
    Tool,
    Function,
    Parameter,
    Role,
    ContentType,
    Reasoning,
    Other,
}

#[derive(Debug, Clone)]
pub enum MatchRule {
    Exact(String),
    StartsWith(String),
    Contains(String),
    And(Box<MatchRule>, Box<MatchRule>),
    Or(Box<MatchRule>, Box<MatchRule>),
    Not(Box<MatchRule>),
}

impl MatchRule {
    pub fn matches(&self, content: &str) -> bool {
        match self {
            MatchRule::Exact(s) => s == content,
            MatchRule::StartsWith(s) => content.starts_with(s),
            MatchRule::Contains(s) => content.contains(s),
            MatchRule::And(lhs, rhs) => lhs.matches(content) && rhs.matches(content),
            MatchRule::Or(lhs, rhs) => lhs.matches(content) || rhs.matches(content),
            MatchRule::Not(inner) => !inner.matches(content),
        }
    }

    pub fn and(self, other: Self) -> Self {
        MatchRule::And(Box::new(self), Box::new(other))
    }

    pub fn or(self, other: Self) -> Self {
        MatchRule::Or(Box::new(self), Box::new(other))
    }

    pub fn not(self) -> Self {
        MatchRule::Not(Box::new(self))
    }
}

#[derive(Debug, Clone)]
pub struct SpecialTokenMatch {
    pub category: Category,
    pub id: u32,
    pub content: String,
}


#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    eos: Vec<(u32, String)>,
    pad: Vec<(u32, String)>,
    bos: Vec<(u32, String)>,
    sep: Vec<(u32, String)>,
    cls: Vec<(u32, String)>,
    mask: Vec<(u32, String)>,
    tool: Vec<(u32, String)>,
    function: Vec<(u32, String)>,
    parameter: Vec<(u32, String)>,
    role: Vec<(u32, String)>,
    content_type: Vec<(u32, String)>,
    reasoning: Vec<(u32, String)>,
    other: Vec<(u32, String)>,
}

impl SpecialTokens {
    /// Search for tokens by ID or by substring within the token content.
    ///
    /// # Arguments
    /// * `id` - Optional token ID to match exactly.
    /// * `substring` - Optional string to search for within the token content (case-sensitive).
    ///
    /// # Returns
    /// A vector of `Match` structs containing the category, id, and string of all matches.
    pub fn search(&self, id: Option<u32>, substring: Option<&str>) -> Vec<SpecialTokenMatch> {
        let mut results = Vec::new();

        // Helper closure to check a single vector of tokens
        let mut check_tokens = |vec: &[(u32, String)], cat: Category| {
            for (token_id, content) in vec {
                let id_match = match id {
                    Some(target_id) => *token_id == target_id,
                    None => true,
                };

                let content_match = match substring {
                    Some(sub) => content.contains(sub),
                    None => true,
                };

                if id_match && content_match {
                    results.push(SpecialTokenMatch {
                        category: cat,
                        id: *token_id,
                        content: content.clone(),
                    });
                }
            }
        };

        // Iterate through all category vectors
        check_tokens(&self.eos, Category::Eos);
        check_tokens(&self.pad, Category::Pad);
        check_tokens(&self.bos, Category::Bos);
        check_tokens(&self.sep, Category::Sep);
        check_tokens(&self.cls, Category::Cls);
        check_tokens(&self.mask, Category::Mask);
        check_tokens(&self.tool, Category::Tool);
        check_tokens(&self.function, Category::Function);
        check_tokens(&self.parameter, Category::Parameter);
        check_tokens(&self.role, Category::Role);
        check_tokens(&self.content_type, Category::ContentType);
        check_tokens(&self.reasoning, Category::Reasoning);
        check_tokens(&self.other, Category::Other);

        results
    }

    pub fn new(tokenizer: &Tokenizer) -> Self {
        let rules = default_rules();
        Self::from_tokenizer_and_rules(tokenizer, &rules)
    }

    pub fn new_from_file(tokenizer_path: &str) -> Self {
        let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");
        Self::new(&tokenizer)
    }

    pub fn from_tokenizer_and_rules(tokenizer: &Tokenizer, rules: &[(MatchRule, Category)]) -> Self {
        let decoder = tokenizer.get_added_tokens_decoder();
        let mut map: std::collections::HashMap<Category, Vec<(u32, String)>> =
            std::collections::HashMap::from([
                (Category::Eos, Vec::new()),
                (Category::Pad, Vec::new()),
                (Category::Bos, Vec::new()),
                (Category::Sep, Vec::new()),
                (Category::Cls, Vec::new()),
                (Category::Mask, Vec::new()),
                (Category::Tool, Vec::new()),
                (Category::Function, Vec::new()),
                (Category::Parameter, Vec::new()),
                (Category::Role, Vec::new()),
                (Category::ContentType, Vec::new()),
                (Category::Reasoning, Vec::new()),
                (Category::Other, Vec::new()),
            ]);
    for (id, AddedToken { content, .. }) in decoder {
        let token = (id, content.clone());

        // 1. Find the first matching rule
        let category = rules.iter()
            .find(|(rule, _)| rule.matches(&content))
            .map(|(_, cat)| *cat)
            .unwrap_or(Category::Other);

        // 2. Get the vector for this category
        let vec = map.get_mut(&category).unwrap();

        // 3. Idiomatic Uniqueness Check: Push only if ID is not already present
        // We assume uniqueness is based on the Token ID (u32)
        if !vec.iter().any(|(existing_id, _)| *existing_id == id) {
            vec.push(token);
        }
    }
        Self {
            eos: map.remove(&Category::Eos).unwrap_or_default(),
            pad: map.remove(&Category::Pad).unwrap_or_default(),
            bos: map.remove(&Category::Bos).unwrap_or_default(),
            sep: map.remove(&Category::Sep).unwrap_or_default(),
            cls: map.remove(&Category::Cls).unwrap_or_default(),
            mask: map.remove(&Category::Mask).unwrap_or_default(),
            tool: map.remove(&Category::Tool).unwrap_or_default(),
            function: map.remove(&Category::Function).unwrap_or_default(),
            parameter: map.remove(&Category::Parameter).unwrap_or_default(),
            role: map.remove(&Category::Role).unwrap_or_default(),
            content_type: map.remove(&Category::ContentType).unwrap_or_default(),
            reasoning: map.remove(&Category::Reasoning).unwrap_or_default(),
            other: map.remove(&Category::Other).unwrap_or_default(),
        }
    }

    pub fn eos_tokens(&self) -> &[(u32, String)] { &self.eos }
    pub fn eos_ids(&self) -> Vec<u32> { self.eos.iter().map(|(id, _)| *id).collect() }
    pub fn eos_strings(&self) -> Vec<String> { self.eos.iter().map(|(_, s)| s.clone()).collect() }

    pub fn pad_tokens(&self) -> &[(u32, String)] { &self.pad }
    pub fn pad_ids(&self) -> Vec<u32> { self.pad.iter().map(|(id, _)| *id).collect() }
    pub fn pad_strings(&self) -> Vec<String> { self.pad.iter().map(|(_, s)| s.clone()).collect() }

    pub fn bos_tokens(&self) -> &[(u32, String)] { &self.bos }
    pub fn bos_ids(&self) -> Vec<u32> { self.bos.iter().map(|(id, _)| *id).collect() }
    pub fn bos_strings(&self) -> Vec<String> { self.bos.iter().map(|(_, s)| s.clone()).collect() }

    pub fn sep_tokens(&self) -> &[(u32, String)] { &self.sep }
    pub fn sep_ids(&self) -> Vec<u32> { self.sep.iter().map(|(id, _)| *id).collect() }
    pub fn sep_strings(&self) -> Vec<String> { self.sep.iter().map(|(_, s)| s.clone()).collect() }

    pub fn cls_tokens(&self) -> &[(u32, String)] { &self.cls }
    pub fn cls_ids(&self) -> Vec<u32> { self.cls.iter().map(|(id, _)| *id).collect() }
    pub fn cls_strings(&self) -> Vec<String> { self.cls.iter().map(|(_, s)| s.clone()).collect() }

    pub fn mask_tokens(&self) -> &[(u32, String)] { &self.mask }
    pub fn mask_ids(&self) -> Vec<u32> { self.mask.iter().map(|(id, _)| *id).collect() }
    pub fn mask_strings(&self) -> Vec<String> { self.mask.iter().map(|(_, s)| s.clone()).collect() }

    pub fn tool_tokens(&self) -> &[(u32, String)] { &self.tool }
    pub fn tool_ids(&self) -> Vec<u32> { self.tool.iter().map(|(id, _)| *id).collect() }
    pub fn tool_strings(&self) -> Vec<String> { self.tool.iter().map(|(_, s)| s.clone()).collect() }

    pub fn function_tokens(&self) -> &[(u32, String)] { &self.function }
    pub fn function_ids(&self) -> Vec<u32> { self.function.iter().map(|(id, _)| *id).collect() }
    pub fn function_strings(&self) -> Vec<String> { self.function.iter().map(|(_, s)| s.clone()).collect() }

    pub fn parameter_tokens(&self) -> &[(u32, String)] { &self.parameter }
    pub fn parameter_ids(&self) -> Vec<u32> { self.parameter.iter().map(|(id, _)| *id).collect() }
    pub fn parameter_strings(&self) -> Vec<String> { self.parameter.iter().map(|(_, s)| s.clone()).collect() }

    pub fn role_tokens(&self) -> &[(u32, String)] { &self.role }
    pub fn role_ids(&self) -> Vec<u32> { self.role.iter().map(|(id, _)| *id).collect() }
    pub fn role_strings(&self) -> Vec<String> { self.role.iter().map(|(_, s)| s.clone()).collect() }

    pub fn content_type_tokens(&self) -> &[(u32, String)] { &self.content_type }
    pub fn content_type_ids(&self) -> Vec<u32> { self.content_type.iter().map(|(id, _)| *id).collect() }
    pub fn content_type_strings(&self) -> Vec<String> { self.content_type.iter().map(|(_, s)| s.clone()).collect() }

    pub fn reasoning_tokens(&self) -> &[(u32, String)] { &self.reasoning }
    pub fn reasoning_ids(&self) -> Vec<u32> { self.reasoning.iter().map(|(id, _)| *id).collect() }
    pub fn reasoning_strings(&self) -> Vec<String> { self.reasoning.iter().map(|(_, s)| s.clone()).collect() }

    pub fn other_tokens(&self) -> &[(u32, String)] { &self.other }
    pub fn other_ids(&self) -> Vec<u32> { self.other.iter().map(|(id, _)| *id).collect() }
    pub fn other_strings(&self) -> Vec<String> { self.other.iter().map(|(_, s)| s.clone()).collect() }
}

pub fn default_rules() -> Vec<(MatchRule, Category)> {
    vec![
        // eos
        (MatchRule::Exact("</s>".to_string()), Category::Eos),
        (MatchRule::Exact("<|end_of_text|>".to_string()), Category::Eos),
        (MatchRule::Exact("<‌|im_end|>".to_string()), Category::Eos),
        (MatchRule::Exact("<eos>".to_string()), Category::Eos),
        (MatchRule::Exact("eos".to_string()), Category::Eos),
        (MatchRule::StartsWith("<|end".to_string()), Category::Eos),
        (MatchRule::StartsWith("<|eod".to_string()), Category::Eos),
        (MatchRule::Contains("end_of".to_string()), Category::Eos),
        (MatchRule::Contains("end".to_string())
            .and(MatchRule::Not(Box::new(MatchRule::Contains("tokenizer".to_string())))),
         Category::Eos),

        // pad
        (MatchRule::Exact("<pad>".to_string()), Category::Pad),
        (MatchRule::Exact("<pad_token>".to_string()), Category::Pad),
        (MatchRule::Exact("pad".to_string()), Category::Pad),
        (MatchRule::Exact("<|video_pad|>".to_string()), Category::Pad),
        (MatchRule::Exact("<|vision_pad|>".to_string()), Category::Pad),
        (MatchRule::Exact("<|fim_pad|>".to_string()), Category::Pad),
        (MatchRule::Exact("<|fim_prefix|>".to_string()), Category::Pad),
        (MatchRule::Exact("<|fim_pad|>".to_string()), Category::Pad),
        (MatchRule::Exact("<|fim_suffix|>".to_string()), Category::Pad),
        (MatchRule::Exact("<|fim_middle|>".to_string()), Category::Pad),
        (MatchRule::Exact("<|image_pad|>".to_string()), Category::Pad),
        (MatchRule::StartsWith("<pad".to_string()), Category::Pad),
        (MatchRule::StartsWith("pad".to_string()), Category::Pad),
        // bos
        (MatchRule::Exact("<s>".to_string()), Category::Bos),
        (MatchRule::Exact("<|start_of_turn|>".to_string()), Category::Bos),
        (MatchRule::Exact("<|vision_start|>".to_string()), Category::Bos),
        (MatchRule::Exact("<|im_start|>".to_string()), Category::Bos),
        (MatchRule::Exact("<|quad_start|>".to_string()), Category::Bos),
        (MatchRule::Exact("<|box_start|>".to_string()), Category::Bos),
        (MatchRule::Exact("<|vision_start|>".to_string()), Category::Bos),
        (MatchRule::Exact("<|start_of_turn|>".to_string()), Category::Bos),
        (MatchRule::Exact("<|object_ref_start|>".to_string()), Category::Bos),
        (MatchRule::StartsWith("<|start".to_string()), Category::Bos),
        (MatchRule::Exact("<|im_start|>".to_string()), Category::Eos),
        (MatchRule::StartsWith("<bos".to_string()), Category::Bos),
        (MatchRule::Exact("bos".to_string()), Category::Bos),
        // sep
        (MatchRule::Exact("<sep>".to_string()), Category::Sep),
        (MatchRule::Exact("<|separator|>".to_string()), Category::Sep),
        (MatchRule::Exact("sep".to_string()), Category::Sep),
        (MatchRule::StartsWith("<sep".to_string()), Category::Sep),
        // cls
        (MatchRule::Exact("<cls>".to_string()), Category::Cls),
        (MatchRule::Exact("[CLS]".to_string()), Category::Cls),
        (MatchRule::Exact("cls".to_string()), Category::Cls),
        (MatchRule::StartsWith("<cls".to_string()), Category::Cls),
        // mask
        (MatchRule::Exact("<mask>".to_string()), Category::Mask),
        (MatchRule::Exact("<mask_token>".to_string()), Category::Mask),
        (MatchRule::Exact("[MASK]".to_string()), Category::Mask),
        (MatchRule::Exact("mask".to_string()), Category::Mask),
        (MatchRule::StartsWith("<mask".to_string()), Category::Mask),
        // tool
        (MatchRule::Exact("<tool>".to_string()), Category::Tool),
        (MatchRule::Exact("<|tool|>".to_string()), Category::Tool),
        (MatchRule::StartsWith("<tool".to_string()), Category::Tool),
        (MatchRule::Contains("<|tool".to_string()), Category::Tool),
        (MatchRule::Contains("tool".to_string()), Category::Tool),
        // function
        (MatchRule::Exact("<function>".to_string()), Category::Function),
        (MatchRule::Exact("<|function|>".to_string()), Category::Function),
        (MatchRule::StartsWith("<function".to_string()), Category::Function),
        (MatchRule::Contains("<|function".to_string()), Category::Function),
        (MatchRule::Contains("function".to_string()), Category::Function),
        // parameter
        (MatchRule::Exact("<parameter>".to_string()), Category::Parameter),
        (MatchRule::Exact("<|parameter|>".to_string()), Category::Parameter),
        (MatchRule::StartsWith("<parameter".to_string()), Category::Parameter),
        (MatchRule::Contains("<|parameter".to_string()), Category::Parameter),
        (MatchRule::Contains("parameter".to_string()), Category::Parameter),
        // role
        (MatchRule::Exact("<role>".to_string()), Category::Role),
        (MatchRule::Exact("<|role|>".to_string()), Category::Role),
        (MatchRule::Exact("<|vision_start|>".to_string()), Category::Role),
        (MatchRule::Exact("<|im_start|>".to_string()), Category::Role),
        (MatchRule::Exact("<|quad_start|>".to_string()), Category::Role),
        (MatchRule::Exact("<|box_start|>".to_string()), Category::Role),
        (MatchRule::Exact("<|vision_start|>".to_string()), Category::Role),
        (MatchRule::Exact("<|file_sep|>".to_string()), Category::Role),
        (MatchRule::Exact("<|im_end|>".to_string()), Category::Role),
        (MatchRule::StartsWith("<|role|>".to_string()), Category::Role),
        (MatchRule::StartsWith("<role".to_string()), Category::Role),
        (MatchRule::Contains("role".to_string()), Category::Role),
        // content_type
        (MatchRule::Exact("<content_type>".to_string()), Category::ContentType),
        (MatchRule::Exact("<|content_type|>".to_string()), Category::ContentType),
        (MatchRule::StartsWith("<content_type".to_string()), Category::ContentType),
        (MatchRule::Contains("<|content_type".to_string()), Category::ContentType),
        (MatchRule::Contains("content_type".to_string()), Category::ContentType),
        // reasoning
        (MatchRule::Exact("<think>".to_string()), Category::Reasoning),
        (MatchRule::Exact("</think>".to_string()), Category::Reasoning),
        (MatchRule::Exact("<thinking>".to_string()), Category::Reasoning),
        (MatchRule::Exact("</thinking>".to_string()), Category::Reasoning),
        (MatchRule::Exact("<reasoning>".to_string()), Category::Reasoning),
        (MatchRule::Exact("<|thinking|>".to_string()), Category::Reasoning),
        (MatchRule::Exact("<|reasoning|>".to_string()), Category::Reasoning),
        (MatchRule::StartsWith("<thinking".to_string()), Category::Reasoning),
        (MatchRule::StartsWith("<reasoning".to_string()), Category::Reasoning),
        (MatchRule::StartsWith("<|think".to_string()), Category::Reasoning),
        (MatchRule::StartsWith("<|reason".to_string()), Category::Reasoning),
        (MatchRule::Contains("thinking".to_string()), Category::Reasoning),
        (MatchRule::Contains("reasoning".to_string()), Category::Reasoning),
        (MatchRule::Contains("chain_of_thought".to_string()), Category::Reasoning),
        (MatchRule::Contains("cot".to_string()), Category::Reasoning),
    ]
}