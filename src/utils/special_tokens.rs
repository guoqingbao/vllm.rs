use std::collections::HashSet;
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

impl Category {
    pub fn search_strings(&self) -> Vec<String> {
        match self {
            Self::Eos => vec![
                    "</s>" , "<eos>" , "<,eos,>" , "eos" , "<,end_of_text,>" , "<,end,>" ,
                    "<,eot,>" , "<,eot_id,>" , "<,eom_id,>" , "<,end_of_turn,>" ,
                    "<,endoftext,>" , "<,endofsequence,>" , "[EOS]"
                ].iter().map(|e| e.to_string()).collect(),
            Self::Bos => vec![
                    "<s>" , "<bos>" , "<,bos,>" , "bos" , "<,bos_token,>" ,
                    "<,begin_of_text,>" , "<,startoftext,>" , "<,start,>" ,
                    "<,im_start,>" , "[BOS]"
                ].iter().map(|e| e.to_string()).collect(),
            Self::Pad => vec![
                    "<pad>" , "<,pad,>" , "<pad_token>" , "<,pad_token,>" ,
                    "pad" , "[PAD]" , "<padding>"
                ].iter().map(|e| e.to_string()).collect(),
            Self::Sep => vec![
                    "<sep>" , "<,sep,>" , "<,separator,>" , "[SEP]"
                ].iter().map(|e| e.to_string()).collect(),
            Self::Cls => vec![
                    "<cls>" , "<,cls,>" , "[CLS]" , "<CLS>"
                ].iter().map(|e| e.to_string()).collect(),
            Self::Mask => vec![
                    "<mask>" , "<,mask,>" , "[MASK]" , "<mask_token>" ,
                    "<,mask_token,>" , "<,infill_mask,>" , "<,extra_id_0,>" ,
                    "<extra_id_0>" , "<extra_id_1>"
                ].iter().map(|e| e.to_string()).collect(),
            Self::Role => vec![
                    "<,system,>" , "<,user,>" , "<,assistant,>" , "<,role,>" ,
                    "<,critic,>" , "<,observer,>" , "<system>" ,
                    "<user>" , "<assistant>" , "<role>"
                ].iter().map(|e| e.to_string()).collect(),
            Self::ContentType => vec![
                    "<,content_type,>" , "<,content,>" , "<,text,>" , "<,code,>" ,
                    "<,json,>" , "<,markdown,>" , "<,output,>" , "<,html,>" ,
                    "<,data,>" , "<,datatype,>"
                ].iter().map(|e| e.to_string()).collect(),
            Self::Tool=> vec![
                    "<|python_tag|>", "<|eom_id|>",
                    "<‌tool_call>","<‌/tool_call>",
                    "[TOOL_CALLS]", "]",
                    "<start_function_call>", "<end_function_call>",
                ].iter().map(|e| e.to_string()).collect(),
            Self::Function => vec![
                    "<,function,>" , "<function>" , "<,functions,>" , "<,fn,>" , "<fn>" ,
                    "<,tool,>" , "<tool>" , "<,tools,>" , "<,api,>" ,
                    "<,invoke,>" , "<,function_call,>" , "<,tool_call,>" ,
                    "<,function_call_json,>"
                ].iter().map(|e| e.to_string()).collect(),
            Self::Parameter => vec![
                    "<parameter>" , "<,parameter,>" , "<,parameters,>" ,
                    "<,args,>" , "<,arguments,>" , "<arguments>" ,
                    "<params>" , "<,params,>"
                ].iter().map(|e| e.to_string()).collect(),
            Self::Reasoning => vec![
                    " magnesium " , " magnesia " , "<thinking>" , "</thinking>" ,
                    "<,thinking,>" , "<,reasoning,>" , "<reasoning>" , "<,reason,>" ,
                    "<,thought,>" , "<,thoughts,>" , "<,internal,>" ,
                    "<internal>" , "</internal>" , "<,reflect,>" ,
                    "<reflection>" , "<,chain_of_thought,>" , "<,analysis,>" ,
                    "<,rationale,>" , "<,explanation,>"
                ].iter().map(|e| e.to_string()).collect(),
            Self::Other => vec![
                    "<,eos_token,>" , "<,unk_token,>" , "<unk>" , "[UNK]" ,
                    "<,start_header_id,>" , "<,end_header_id,>" ,
                    "<,metadata,>" , "<,special,>"
                ].iter().map(|e| e.to_string()).collect(),
        }
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
                content: added_token.content.clone().into_bytes(),
                source,
            });
        }

        // Step 2: Add tokens from base vocabulary that match known patterns
        let vocab = tokenizer.get_vocab(true);
        for (token_str, id) in vocab {
            if seen_ids.contains(&id) {
                continue; // Already processed as added token
            }

            // Try to categorize by content
            let category = Self::categorize_by_content(token_str.as_str());
            if category != Category::Other {
                token_set.push(SpecialToken {
                    category,
                    id,
                    content: token_str.as_bytes().to_vec(),
                    source: VocabSource::Common,
                });
            }
        }

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

    /// Get tool start token IDs (tokens containing 'tool_call' or 'function_call' without closing slash)
    pub fn tool_start_ids(&self) -> Vec<u32> {
        self.tool()
            .iter()
            .filter(|t| {
                let s = t.string();
                (s.contains("tool_call") || s.contains("function_call")) && !s.contains("</")
            })
            .map(|t| t.id)
            .collect()
    }

    /// Get tool end token IDs (tokens containing '</tool_call' or '</function_call')
    pub fn tool_end_ids(&self) -> Vec<u32> {
        self.tool()
            .iter()
            .filter(|t| {
                let s = t.string();
                s.contains("</tool_call") || s.contains("</function_call")
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
