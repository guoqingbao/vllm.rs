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
pub struct SpecialTokenMatch {
    pub category: Category,
    pub id: u32,
    pub content: String,
}

#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    eos: Vec<(u32, AddedToken)>,
    pad: Vec<(u32, AddedToken)>,
    bos: Vec<(u32, AddedToken)>,
    sep: Vec<(u32, AddedToken)>,
    cls: Vec<(u32, AddedToken)>,
    mask: Vec<(u32, AddedToken)>,
    tool: Vec<(u32, AddedToken)>,
    function: Vec<(u32, AddedToken)>,
    parameter: Vec<(u32, AddedToken)>,
    role: Vec<(u32, AddedToken)>,
    content_type: Vec<(u32, AddedToken)>,
    reasoning: Vec<(u32, AddedToken)>,
    other: Vec<(u32, AddedToken)>,
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
        let mut check_tokens = |vec: &[(u32, AddedToken)], cat: Category| {
            for (token_id, added_token) in vec {
                let content = &added_token.content;
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
        let decoder = tokenizer.get_added_tokens_decoder();
        let mut map: std::collections::HashMap<Category, Vec<(u32, AddedToken)>> =
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
        
        for (id, added_token) in decoder {
            let token = (id, added_token);

            // Use the `special` field from AddedToken to categorize
            let category = if token.1.special {
                // Special tokens - categorize by content patterns
                match token.1.content.as_str() {
                    // End of sequence / text
                    "</s>" | "<eos>" | "<|eos|>" | "eos" | "<|end_of_text|>" | "<|end|>" |
                    "<|eot|>" | "<|eot_id|>" | "<|eom_id|>" | "<|end_of_turn|>" |
                    "<|endoftext|>" | "<|endofsequence|>" | "[EOS]" => Category::Eos,

                    // Beginning of sequence / text
                    "<s>" | "<bos>" | "<|bos|>" | "bos" | "<|bos_token|>" |
                    "<|begin_of_text|>" | "<|startoftext|>" | "<|start|>" |
                    "<|im_start|>" | "[BOS]" => Category::Bos,

                    // Padding
                    "<pad>" | "<|pad|>" | "<pad_token>" | "<|pad_token|>" |
                    "pad" | "[PAD]" | "<padding>" => Category::Pad,

                    // Separator
                    "<sep>" | "<|sep|>" | "<|separator|>" | "[SEP]" => Category::Sep,

                    // Classification
                    "<cls>" | "<|cls|>" | "[CLS]" | "<CLS>" => Category::Cls,

                    // Mask / Infill
                    "<mask>" | "<|mask|>" | "[MASK]" | "<mask_token>" |
                    "<|mask_token|>" | "<|infill_mask|>" | "<|extra_id_0|>" |
                    "<extra_id_0>" | "<extra_id_1>" => Category::Mask,

                    // Role / Conversation markers
                    "<|system|>" | "<|user|>" | "<|assistant|>" | "<|role|>" |
                    "<|critic|>" | "<|observer|>" | "<system>" |
                    "<user>" | "<assistant>" | "<role>" => Category::Role,

                    // Content typing
                    "<|content_type|>" | "<|content|>" | "<|text|>" | "<|code|>" |
                    "<|json|>" | "<|markdown|>" | "<|output|>" | "<|html|>" |
                    "<|data|>" | "<|datatype|>" => Category::ContentType,

                    // Function / Tool invocation
                    "<|function|>" | "<function>" | "<|functions|>" | "<|fn|>" | "<fn>" |
                    "<|tool|>" | "<tool>" | "<|tools|>" | "<|api|>" |
                    "<|invoke|>" | "<|function_call|>" | "<|tool_call|>" |
                    "<|function_call_json|>" => Category::Function,

                    // Parameter / Argument delimiters
                    "<parameter>" | "<|parameter|>" | "<|parameters|>" |
                    "<|args|>" | "<|arguments|>" | "<arguments>" |
                    "<params>" | "<|params|>" => Category::Parameter,

                    // Reasoning / Thinking / Reflection
                    "<think>" | "</think>" | "<thinking>" | "</thinking>" |
                    "<|thinking|>" | "<|reasoning|>" | "<reasoning>" | "<|reason|>" |
                    "<|thought|>" | "<|thoughts|>" | "<|internal|>" |
                    "<internal>" | "</internal>" | "<|reflect|>" |
                    "<reflection>" | "<|chain_of_thought|>" | "<|analysis|>" |
                    "<|rationale|>" | "<|explanation|>" => Category::Reasoning,

                    // Control / Non-semantic system tags
                    "<|eos_token|>" | "<|unk_token|>" | "<unk>" | "[UNK]" |
                    "<|start_header_id|>" | "<|end_header_id|>" |
                    "<|metadata|>" | "<|special|>" => Category::Other,

                    _ => Category::Other,
                }


            } else {
                Category::Other
            };

            // Push only if ID is not already present
            let vec = map.get_mut(&category).unwrap();
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

    pub fn new_from_file(tokenizer_path: &str) -> Self {
        let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");
        Self::new(&tokenizer)
    }

    pub fn eos_tokens(&self) -> &[(u32, AddedToken)] { &self.eos }
    pub fn eos_ids(&self) -> Vec<u32> { self.eos.iter().map(|(id, _)| *id).collect() }
    pub fn eos_strings(&self) -> Vec<String> { self.eos.iter().map(|(_, t)| t.content.clone()).collect() }
    pub fn eos_special(&self) -> Vec<bool> { self.eos.iter().map(|(_, t)| t.special).collect() }

    pub fn pad_tokens(&self) -> &[(u32, AddedToken)] { &self.pad }
    pub fn pad_ids(&self) -> Vec<u32> { self.pad.iter().map(|(id, _)| *id).collect() }
    pub fn pad_strings(&self) -> Vec<String> { self.pad.iter().map(|(_, t)| t.content.clone()).collect() }
    pub fn pad_special(&self) -> Vec<bool> { self.pad.iter().map(|(_, t)| t.special).collect() }

    pub fn bos_tokens(&self) -> &[(u32, AddedToken)] { &self.bos }
    pub fn bos_ids(&self) -> Vec<u32> { self.bos.iter().map(|(id, _)| *id).collect() }
    pub fn bos_strings(&self) -> Vec<String> { self.bos.iter().map(|(_, t)| t.content.clone()).collect() }
    pub fn bos_special(&self) -> Vec<bool> { self.bos.iter().map(|(_, t)| t.special).collect() }

    pub fn sep_tokens(&self) -> &[(u32, AddedToken)] { &self.sep }
    pub fn sep_ids(&self) -> Vec<u32> { self.sep.iter().map(|(id, _)| *id).collect() }
    pub fn sep_strings(&self) -> Vec<String> { self.sep.iter().map(|(_, t)| t.content.clone()).collect() }
    pub fn sep_special(&self) -> Vec<bool> { self.sep.iter().map(|(_, t)| t.special).collect() }

    pub fn cls_tokens(&self) -> &[(u32, AddedToken)] { &self.cls }
    pub fn cls_ids(&self) -> Vec<u32> { self.cls.iter().map(|(id, _)| *id).collect() }
    pub fn cls_strings(&self) -> Vec<String> { self.cls.iter().map(|(_, t)| t.content.clone()).collect() }
    pub fn cls_special(&self) -> Vec<bool> { self.cls.iter().map(|(_, t)| t.special).collect() }

    pub fn mask_tokens(&self) -> &[(u32, AddedToken)] { &self.mask }
    pub fn mask_ids(&self) -> Vec<u32> { self.mask.iter().map(|(id, _)| *id).collect() }
    pub fn mask_strings(&self) -> Vec<String> { self.mask.iter().map(|(_, t)| t.content.clone()).collect() }
    pub fn mask_special(&self) -> Vec<bool> { self.mask.iter().map(|(_, t)| t.special).collect() }

    pub fn tool_tokens(&self) -> &[(u32, AddedToken)] { &self.tool }
    pub fn tool_ids(&self) -> Vec<u32> { self.tool.iter().map(|(id, _)| *id).collect() }
    pub fn tool_strings(&self) -> Vec<String> { self.tool.iter().map(|(_, t)| t.content.clone()).collect() }
    pub fn tool_special(&self) -> Vec<bool> { self.tool.iter().map(|(_, t)| t.special).collect() }

    pub fn function_tokens(&self) -> &[(u32, AddedToken)] { &self.function }
    pub fn function_ids(&self) -> Vec<u32> { self.function.iter().map(|(id, _)| *id).collect() }
    pub fn function_strings(&self) -> Vec<String> { self.function.iter().map(|(_, t)| t.content.clone()).collect() }
    pub fn function_special(&self) -> Vec<bool> { self.function.iter().map(|(_, t)| t.special).collect() }

    pub fn parameter_tokens(&self) -> &[(u32, AddedToken)] { &self.parameter }
    pub fn parameter_ids(&self) -> Vec<u32> { self.parameter.iter().map(|(id, _)| *id).collect() }
    pub fn parameter_strings(&self) -> Vec<String> { self.parameter.iter().map(|(_, t)| t.content.clone()).collect() }
    pub fn parameter_special(&self) -> Vec<bool> { self.parameter.iter().map(|(_, t)| t.special).collect() }

    pub fn role_tokens(&self) -> &[(u32, AddedToken)] { &self.role }
    pub fn role_ids(&self) -> Vec<u32> { self.role.iter().map(|(id, _)| *id).collect() }
    pub fn role_strings(&self) -> Vec<String> { self.role.iter().map(|(_, t)| t.content.clone()).collect() }
    pub fn role_special(&self) -> Vec<bool> { self.role.iter().map(|(_, t)| t.special).collect() }

    pub fn content_type_tokens(&self) -> &[(u32, AddedToken)] { &self.content_type }
    pub fn content_type_ids(&self) -> Vec<u32> { self.content_type.iter().map(|(id, _)| *id).collect() }
    pub fn content_type_strings(&self) -> Vec<String> { self.content_type.iter().map(|(_, t)| t.content.clone()).collect() }
    pub fn content_type_special(&self) -> Vec<bool> { self.content_type.iter().map(|(_, t)| t.special).collect() }

    pub fn reasoning_tokens(&self) -> &[(u32, AddedToken)] { &self.reasoning }
    pub fn reasoning_ids(&self) -> Vec<u32> { self.reasoning.iter().map(|(id, _)| *id).collect() }
    pub fn reasoning_strings(&self) -> Vec<String> { self.reasoning.iter().map(|(_, t)| t.content.clone()).collect() }
    pub fn reasoning_special(&self) -> Vec<bool> { self.reasoning.iter().map(|(_, t)| t.special).collect() }

    pub fn other_tokens(&self) -> &[(u32, AddedToken)] { &self.other }
    pub fn other_ids(&self) -> Vec<u32> { self.other.iter().map(|(id, _)| *id).collect() }
    pub fn other_strings(&self) -> Vec<String> { self.other.iter().map(|(_, t)| t.content.clone()).collect() }
    pub fn other_special(&self) -> Vec<bool> { self.other.iter().map(|(_, t)| t.special).collect() }

    /// Get all tokens across all categories
    pub fn all_tokens(&self) -> Vec<(u32, AddedToken)> {
        let mut all = Vec::new();
        all.extend(self.eos.iter().cloned());
        all.extend(self.pad.iter().cloned());
        all.extend(self.bos.iter().cloned());
        all.extend(self.sep.iter().cloned());
        all.extend(self.cls.iter().cloned());
        all.extend(self.mask.iter().cloned());
        all.extend(self.tool.iter().cloned());
        all.extend(self.function.iter().cloned());
        all.extend(self.parameter.iter().cloned());
        all.extend(self.role.iter().cloned());
        all.extend(self.content_type.iter().cloned());
        all.extend(self.reasoning.iter().cloned());
        all.extend(self.other.iter().cloned());
        all
    }

    /// Get all special tokens (where AddedToken.special == true)
    pub fn all_special(&self) -> Vec<(u32, AddedToken)> {
        self.all_tokens().into_iter().filter(|(_, t)| t.special).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_accessors() {
        // Test that AddedToken fields are accessible
        let added_token = AddedToken {
            content: "<test>".to_string(),
            special: true,
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
        };
        
        assert_eq!(added_token.content, "<test>");
        assert!(added_token.special);
    }

    #[test]
    fn test_categorize_special_tokens() {
        // Test that special tokens are correctly categorized
        let tokenizer = Tokenizer::from_file("tests/fixtures/tokenizer.json").ok();
        
        if let Some(tok) = tokenizer {
            let special_tokens = SpecialTokens::new(&tok);
            
            // Check that we have some tokens stored
            assert!(!special_tokens.eos_tokens().is_empty() || 
                    !special_tokens.pad_tokens().is_empty());
        }
    }

    #[test]
    fn test_search_by_id() {
        let tokenizer = Tokenizer::from_file("tests/fixtures/tokenizer.json").ok();
        
        if let Some(tok) = tokenizer {
            let special_tokens = SpecialTokens::new(&tok);
            
            // Search for a specific ID
            let results = special_tokens.search(Some(2), None);
            
            // Each result should have the matching ID
            for result in &results {
                assert_eq!(result.id, 2);
            }
        }
    }

    #[test]
    fn test_search_by_content() {
        let tokenizer = Tokenizer::from_file("tests/fixtures/tokenizer.json").ok();
        
        if let Some(tok) = tokenizer {
            let special_tokens = SpecialTokens::new(&tok);
            
            // Search for tokens containing "end"
            let results = special_tokens.search(None, Some("end"));
            
            // Each result should contain the search string
            for result in &results {
                assert!(result.content.contains("end"));
            }
        }
    }

    #[test]
    fn test_token_uniqueness() {
        let tokenizer = Tokenizer::from_file("tests/fixtures/tokenizer.json").ok();
        
        if let Some(tok) = tokenizer {
            let special_tokens = SpecialTokens::new(&tok);
            
            // Check that no category has duplicate IDs
            let all_ids: Vec<u32> = special_tokens.eos_ids()
                .into_iter()
                .chain(special_tokens.pad_ids())
                .chain(special_tokens.bos_ids())
                .collect();
            
            let unique_ids: std::collections::HashSet<u32> = all_ids.iter().cloned().collect();
            assert_eq!(all_ids.len(), unique_ids.len(), "Duplicate token IDs found");
        }
    }
}
