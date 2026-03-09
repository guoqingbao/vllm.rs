use std::collections::HashSet;
use image::EncodableLayout;
use tokenizers::tokenizer::Tokenizer;

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


impl Category {
    pub fn search_strings(&self) -> Vec<String> {
        match self {
            Self::Eos => vec![
                    "</s>" , "<eos>" , "<,eos,>", "<,end_of_text,>" , "<,end,>" ,
                    "<,eot,>" , "<,eot_id,>" , "<,eom_id,>" , "<,end_of_turn,>" ,
                    "<,endoftext,>" , "<,endofsequence,>" , "[EOS]", "<|im_end|>",
                    "<|box_end|>", "<|object_ref_end|>","<|quad_end|>", "<|endoftext|>",
                    "<|vision_end|>", "<|eot|>", "<|python_end|>", "<|end_of_text|>",
                    "<|header_end|>", "<|eom|>"
                ],
            Self::Bos => vec![
                    "<s>", "<bos>" , "<,bos,>", "<,bos_token,>" ,
                    "<,begin_of_text,>" , "<,startoftext,>" , "<,start,>" ,
                    "<,im_start,>" , "[BOS]", "<|box_start|>", "<|im_start|>",
                    "<|object_ref_start|>", "<|quad_start|>", "<|vision_start|>",
                    "<|python_start|>", "<|begin_of_text|>", "<|header_start|>"
                ],
            Self::Pad => vec![
                    "<pad>" , "<,pad,>" , "<pad_token>" , "<,pad_token,>" ,
                    "[PAD]" , "<padding>", "<|image_pad|>", "<|video_pad|>",
                    "<|vision_pad|>",
                ],
            Self::Sep => vec![
                    "<sep>" , "<,sep,>" , "<,separator,>" , "[SEP]"
                ],
            Self::Cls => vec![
                    "<cls>" , "<,cls,>" , "[CLS]" , "<CLS>"
                ],
            Self::Mask => vec![
                    "<mask>" , "<,mask,>" , "[MASK]" , "<mask_token>" ,
                    "<,mask_token,>" , "<,infill_mask,>" , "<,extra_id_0,>" ,
                    "<extra_id_0>" , "<extra_id_1>"
                ],
            Self::Role => vec![
                    "<,system,>" , "<,user,>" , "<,assistant,>" , "<,role,>" ,
                    "<,critic,>" , "<,observer,>" , "<system>" ,
                    "<user>" , "<assistant>" , "<role>"
                ],
            Self::ContentType => vec![
                    "<,content_type,>" , "<,content,>" , "<,text,>" , "<,code,>" ,
                    "<,json,>" , "<,markdown,>" , "<,output,>" , "<,html,>" ,
                    "<,data,>" , "<,datatype,>", "<|image|>"
                ],
            Self::Tool=> vec![
                    "<|python_tag|>", "<|eom_id|>",
                    "<tool_call>", "</tool_call>",
                    "</tool_response>", "<tool_response>",
                    "[TOOL_CALLS]", "]",
                    "<start_function_call>", "<end_function_call>",

                ],
            Self::Function => vec![
                    "<,function,>" , "<function>" , "<,functions,>" , "<,fn,>" , "<fn>" ,
                    "<,tool,>" , "<tool>" , "<,tools,>" , "<,api,>" ,
                    "<,invoke,>" , "<,function_call,>" , "<,tool_call,>" ,
                    "<,function_call_json,>"
                ],
            Self::Parameter => vec![
                    "<parameter>" , "<,parameter,>" , "<,parameters,>" ,
                    "<,args,>" , "<,arguments,>" , "<arguments>" ,
                    "<params>" , "<,params,>"
                ],
            Self::Reasoning => vec![
                    " magnesium " , " magnesia " , "<thinking>" , "</thinking>" ,
                    "<,thinking,>" , "<,reasoning,>" , "<reasoning>" , "<,reason,>" ,
                    "<,thought,>" , "<,thoughts,>" , "<,internal,>" ,
                    "<internal>" , "</internal>" , "<,reflect,>" ,
                    "<reflection>" , "<,chain_of_thought,>" , "<,analysis,>" ,
                    "<,rationale,>" , "<,explanation,>", "<think>", "</think>"
                ],
            Self::Other => vec![
                    "<,eos_token,>" , "<,unk_token,>" , "<unk>" , "[UNK]" ,
                    "<,start_header_id,>" , "<,end_header_id,>" ,
                    "<,metadata,>" , "<,special,>"
                ],
        }.iter().map(|e| e.to_string()).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabSource {
    Special,
    Added,
    Common,
}

#[derive(Debug, Clone)]
pub struct SpecialToken {
    pub category: Category,
    pub id: u32,
    pub content: Vec<u8>,
    pub source: VocabSource,
    pub normalized: bool,
}

impl SpecialToken {
    pub fn string(&self) -> String {
        self.content.clone()
        .into_iter()
        .filter(|b| b.is_ascii())
        .map(|b| b as char)
        .collect()
    }
}


#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    token_set: Vec<SpecialToken>,
}

// Private macros for internal implementation
macro_rules! filter_by_category {
    ($self:ident, $cat:ident) => {
        $self.token_set.iter().filter(|t| t.category == Category::$cat).cloned().collect::<Vec<_>>()
    };
}

macro_rules! filter_by_category_source {
    ($self:ident, $cat:ident, $src:ident) => {
        $self.token_set.iter()
            .filter(|t| t.category == Category::$cat && t.source == VocabSource::$src)
            .cloned()
            .collect::<Vec<_>>()
    };
}

impl SpecialTokens {
    // Public category accessors
    pub fn eos(&self) -> Vec<SpecialToken> { filter_by_category!(self, Eos) }
    pub fn pad(&self) -> Vec<SpecialToken> { filter_by_category!(self, Pad) }
    pub fn bos(&self) -> Vec<SpecialToken> { filter_by_category!(self, Bos) }
    pub fn sep(&self) -> Vec<SpecialToken> { filter_by_category!(self, Sep) }
    pub fn cls(&self) -> Vec<SpecialToken> { filter_by_category!(self, Cls) }
    pub fn mask(&self) -> Vec<SpecialToken> { filter_by_category!(self, Mask) }
    pub fn tool(&self) -> Vec<SpecialToken> { filter_by_category!(self, Tool) }
    pub fn function(&self) -> Vec<SpecialToken> { filter_by_category!(self, Function) }
    pub fn parameter(&self) -> Vec<SpecialToken> { filter_by_category!(self, Parameter) }
    pub fn role(&self) -> Vec<SpecialToken> { filter_by_category!(self, Role) }
    pub fn content_type(&self) -> Vec<SpecialToken> { filter_by_category!(self, ContentType) }
    pub fn reasoning(&self) -> Vec<SpecialToken> { filter_by_category!(self, Reasoning) }
    pub fn other(&self) -> Vec<SpecialToken> { filter_by_category!(self, Other) }

    // Public ID accessors returning Vec<u32>
    pub fn eos_ids(&self) -> Vec<u32> { self.eos().iter().map(|t| t.id).collect() }
    pub fn pad_ids(&self) -> Vec<u32> { self.pad().iter().map(|t| t.id).collect() }
    pub fn bos_ids(&self) -> Vec<u32> { self.bos().iter().map(|t| t.id).collect() }
    pub fn sep_ids(&self) -> Vec<u32> { self.sep().iter().map(|t| t.id).collect() }
    pub fn cls_ids(&self) -> Vec<u32> { self.cls().iter().map(|t| t.id).collect() }
    pub fn mask_ids(&self) -> Vec<u32> { self.mask().iter().map(|t| t.id).collect() }
    pub fn tool_ids(&self) -> Vec<u32> { self.tool().iter().map(|t| t.id).collect() }
    pub fn function_ids(&self) -> Vec<u32> { self.function().iter().map(|t| t.id).collect() }
    pub fn parameter_ids(&self) -> Vec<u32> { self.parameter().iter().map(|t| t.id).collect() }
    pub fn role_ids(&self) -> Vec<u32> { self.role().iter().map(|t| t.id).collect() }
    pub fn content_type_ids(&self) -> Vec<u32> { self.content_type().iter().map(|t| t.id).collect() }
    pub fn reasoning_ids(&self) -> Vec<u32> { self.reasoning().iter().map(|t| t.id).collect() }
    pub fn other_ids(&self) -> Vec<u32> { self.other().iter().map(|t| t.id).collect() }

    // Public ID accessors returning HashSet<u32> for O(1) lookup
    pub fn eos_ids_set(&self) -> HashSet<u32> { self.eos_ids().into_iter().collect() }
    pub fn pad_ids_set(&self) -> HashSet<u32> { self.pad_ids().into_iter().collect() }
    pub fn bos_ids_set(&self) -> HashSet<u32> { self.bos_ids().into_iter().collect() }
    pub fn sep_ids_set(&self) -> HashSet<u32> { self.sep_ids().into_iter().collect() }
    pub fn cls_ids_set(&self) -> HashSet<u32> { self.cls_ids().into_iter().collect() }
    pub fn mask_ids_set(&self) -> HashSet<u32> { self.mask_ids().into_iter().collect() }
    pub fn tool_ids_set(&self) -> HashSet<u32> { self.tool_ids().into_iter().collect() }
    pub fn function_ids_set(&self) -> HashSet<u32> { self.function_ids().into_iter().collect() }
    pub fn parameter_ids_set(&self) -> HashSet<u32> { self.parameter_ids().into_iter().collect() }
    pub fn role_ids_set(&self) -> HashSet<u32> { self.role_ids().into_iter().collect() }
    pub fn content_type_ids_set(&self) -> HashSet<u32> { self.content_type_ids().into_iter().collect() }
    pub fn reasoning_ids_set(&self) -> HashSet<u32> { self.reasoning_ids().into_iter().collect() }
    pub fn other_ids_set(&self) -> HashSet<u32> { self.other_ids().into_iter().collect() }

    /// Get all token IDs across all categories as HashSet for O(1) lookup
    pub fn all_ids_set(&self) -> HashSet<u32> { self.token_set.iter().map(|t| t.id).collect() }

    // Public string accessors
    pub fn eos_strings(&self) -> Vec<String> { self.eos().iter().map(|t| t.string()).collect() }
    pub fn pad_strings(&self) -> Vec<String> { self.pad().iter().map(|t| t.string()).collect() }
    pub fn bos_strings(&self) -> Vec<String> { self.bos().iter().map(|t| t.string()).collect() }
    pub fn sep_strings(&self) -> Vec<String> { self.sep().iter().map(|t| t.string()).collect() }
    pub fn cls_strings(&self) -> Vec<String> { self.cls().iter().map(|t| t.string()).collect() }
    pub fn mask_strings(&self) -> Vec<String> { self.mask().iter().map(|t| t.string()).collect() }
    pub fn tool_strings(&self) -> Vec<String> { self.tool().iter().map(|t| t.string()).collect() }
    pub fn function_strings(&self) -> Vec<String> { self.function().iter().map(|t| t.string()).collect() }
    pub fn parameter_strings(&self) -> Vec<String> { self.parameter().iter().map(|t| t.string()).collect() }
    pub fn role_strings(&self) -> Vec<String> { self.role().iter().map(|t| t.string()).collect() }
    pub fn content_type_strings(&self) -> Vec<String> { self.content_type().iter().map(|t| t.string()).collect() }
    pub fn reasoning_strings(&self) -> Vec<String> { self.reasoning().iter().map(|t| t.string()).collect() }
    pub fn other_strings(&self) -> Vec<String> { self.other().iter().map(|t| t.string()).collect() }

    /// Search for tokens by ID, substring, category, and source.
    /// All parameters are optional - use None to skip that filter.
    pub fn search(
        &self,
        id: Option<u32>,
        substring: Option<&str>,
        category: Option<Category>,
        source: Option<VocabSource>,
    ) -> Vec<SpecialToken> {
        let mut results = Vec::new();
        
        for token in &self.token_set {
            // Filter by ID if specified
            if let Some(target_id) = id {
                if token.id != target_id {
                    continue;
                }
            }
            
            // Filter by substring if specified
            if let Some(sub) = substring {
                let token_str = token.string();
                if !token_str.contains(sub) {
                    continue;
                }
            }
            
            // Filter by category if specified
            if let Some(cat) = category {
                if token.category != cat {
                    continue;
                }
            }
            
            // Filter by source if specified
            if let Some(src) = source {
                if token.source != src {
                    continue;
                }
            }
            
            results.push(token.clone());
        }
        
        results
    }

    /// Create SpecialTokens from a tokenizer
    pub fn new(tokenizer: &Tokenizer) -> Self {
        let mut token_set: Vec<SpecialToken> = Vec::new();
        let mut seen_ids: HashSet<u32> = HashSet::new();

        // Step 1: Process all tokens from tokenizer (added + base vocab)
        // First, get added tokens
        for (id, added_token) in tokenizer.get_added_tokens_decoder() {
            if seen_ids.contains(&id) {
                continue;
            }
            seen_ids.insert(id);

            // Determine category from content
            let category = Self::categorize_by_content(&added_token.content);

            // Determine source
            let source = if added_token.special {
                VocabSource::Special
            } else {
                VocabSource::Added
            };

            token_set.push(SpecialToken {
                category,
                id,
                content: added_token.content.as_bytes().to_vec(),
                source,
                normalized: added_token.normalized
            });
        }

        // Step 2: Add tokens from base vocabulary that match known patterns
        let vocab = tokenizer.get_vocab(true);
        for (token_str, id) in vocab {
            // Find potential duplicates of our special tokens in the common vocab
            if seen_ids.contains(&id) || !token_set.iter().any(
                |f| String::from_utf8(f.content.clone()).unwrap() == token_str.to_string()
            ) {
                continue;
            }

            // Try to categorize by content
            let category = Self::categorize_by_content(token_str.as_str());
            if category != Category::Other {
                token_set.push(SpecialToken {
                    category,
                    id,
                    content: token_str.as_bytes().to_vec(),
                    source: VocabSource::Common,
                    normalized: false,
                });
            }
        }

        // Sort by id for consistent ordering
        token_set.sort_by_key(|t| t.id);

        Self { token_set }
    }
    fn categorize_by_content(content: &str) -> Category {
        for cat in &[Category::Eos, Category::Pad, Category::Bos, Category::Sep,
                     Category::Cls, Category::Mask, Category::Tool, Category::Function,
                     Category::Parameter, Category::Role, Category::ContentType, Category::Reasoning] {
            if cat.search_strings().iter().any(|s| s == content) {
                return *cat;
            }
        }
        Category::Other
    }

    /// Create SpecialTokens from a tokenizer file path
    pub fn new_from_file(tokenizer_path: &str) -> Self {
        let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");
        Self::new(&tokenizer)
    }

    /// Get tool start token IDs (tokens categorized as Tool category that are start markers)
    /// Start markers are those that don't start with </ (not closing tags) and don't end with ]
    pub fn tool_start_ids(&self) -> Vec<u32> {
        self.tool()
            .iter()
            .filter(|t| {
                let s = t.string();
                !s.starts_with("</") && !s.ends_with("]")
            })
            .map(|t| t.id)
            .collect()
    }

    /// Get tool end token IDs (tokens categorized as Tool category that are end markers)
    /// End markers either start with </ (XML closing tags) or end with ] (Mistral style)
    pub fn tool_end_ids(&self) -> Vec<u32> {
        self.tool()
            .iter()
            .filter(|t| {
                let s = t.string();
                s.starts_with("</") || s.ends_with("]")
            })
            .map(|t| t.id)
            .collect()
    }

    /// Get tool start token IDs as HashSet for O(1) lookup
    pub fn tool_start_ids_set(&self) -> HashSet<u32> {
        self.tool_start_ids().into_iter().collect()
    }

    /// Get tool end token IDs as HashSet for O(1) lookup
    pub fn tool_end_ids_set(&self) -> HashSet<u32> {
        self.tool_end_ids().into_iter().collect()
    }

    /// Get tool start token SpecialToken if available
    /// Returns the SpecialToken object containing both ID and string representation
    pub fn tool_start_token(&self) -> Option<SpecialToken> {
        self.tool().iter().find(|t| {
            let s = t.string();
            !s.starts_with("</") && !s.ends_with("]")
        }).cloned()
    }

    /// Get tool end token SpecialToken if available
    /// Returns the SpecialToken object containing both ID and string representation
    pub fn tool_end_token(&self) -> Option<SpecialToken> {
        self.tool().iter().find(|t| {
            let s = t.string();
            s.starts_with("</") || s.ends_with("]")
        }).cloned()
    }

    /// Get tool start and end token SpecialTokens as a pair if both available
    /// Returns None if either token is not found, enabling graceful fallback
    pub fn tool_tokens(&self) -> Option<(SpecialToken, SpecialToken)> {
        let start = self.tool_start_token()?;
        let end = self.tool_end_token()?;
        Some((start, end))
    }

    /// Get all tokens
    pub fn all_tokens(&self) -> Vec<SpecialToken> {
        self.token_set.clone()
    }

    /// Get all special tokens (Special or Added source)
    pub fn all_special(&self) -> Vec<SpecialToken> {
        self.token_set.iter()
            .filter(|t| t.source == VocabSource::Special || t.source == VocabSource::Added)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_token_string_conversion() {
        let token = SpecialToken {
            category: Category::Eos,
            id: 2,
            content: b"</s>".to_vec(),
            source: VocabSource::Added,
            normalized: false,
        };
        assert_eq!(token.string(), "</s>");
    }

    #[test]
    fn test_categorize_special_tokens() {
        let tokenizer = Tokenizer::from_file("tests/fixtures/tokenizer.json").ok();
        
        if let Some(tok) = tokenizer {
            let special_tokens = SpecialTokens::new(&tok);
            
            // Check that we have some tokens stored
            assert!(!special_tokens.eos().is_empty() || 
                    !special_tokens.pad().is_empty());
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
            
            let unique_ids: HashSet<u32> = all_ids.iter().cloned().collect();
            assert_eq!(all_ids.len(), unique_ids.len(), "Duplicate token IDs found");
        }
    }

    #[test]
    fn test_search_by_id() {
        let tokenizer = Tokenizer::from_file("tests/fixtures/tokenizer.json").ok();
        
        if let Some(tok) = tokenizer {
            let special_tokens = SpecialTokens::new(&tok);
            
            // Search for a specific ID
            let results = special_tokens.search(Some(2), None, None, None);
            
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
            let results = special_tokens.search(None, Some("end"), None, None);
            
            // Each result should contain the search string
            for result in &results {
                assert!(result.string().contains("end"));
            }
        }
    }

    #[test]
    fn test_search_by_category() {
        let tokenizer = Tokenizer::from_file("tests/fixtures/tokenizer.json").ok();
        
        if let Some(tok) = tokenizer {
            let special_tokens = SpecialTokens::new(&tok);
            
            // Search for EOS tokens
            let results = special_tokens.search(None, None, Some(Category::Eos), None);
            
            // Each result should be an EOS token
            for result in &results {
                assert_eq!(result.category, Category::Eos);
            }
        }
    }

    #[test]
    fn test_search_by_source() {
        let tokenizer = Tokenizer::from_file("tests/fixtures/tokenizer.json").ok();
        
        if let Some(tok) = tokenizer {
            let special_tokens = SpecialTokens::new(&tok);
            
            // Search for tokens with Added source
            let results = special_tokens.search(None, None, None, Some(VocabSource::Added));
            
            // Each result should have Added source
            for result in &results {
                assert_eq!(result.source, VocabSource::Added);
            }
        }
    }

    #[test]
    fn test_combined_search() {
        let tokenizer = Tokenizer::from_file("tests/fixtures/tokenizer.json").ok();
        
        if let Some(tok) = tokenizer {
            let special_tokens = SpecialTokens::new(&tok);
            
            // Search for EOS tokens with specific substring
            let results = special_tokens.search(None, Some("end"), Some(Category::Eos), None);
            
            // Each result should match all criteria
            for result in &results {
                assert_eq!(result.category, Category::Eos);
                assert!(result.string().contains("end"));
            }
        }
    }
}
