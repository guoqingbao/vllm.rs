// src/utils/guidance.rs
use anyhow::Result;
use llguidance::{api::TopLevelGrammar, Matcher, ParserFactory as LlgParserFactory};
use std::sync::Arc;
use tokenizers::Tokenizer;
use toktrie::{SimpleVob, TokTrie};
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

use crate::utils::config::Constraint;

pub struct GuidanceState {
    matcher: Matcher,
}

impl GuidanceState {
    pub fn new(factory: Arc<ParserFactory>, constraint: &Constraint) -> Result<Self> {
        let grammar = llg_grammar_from_constraint(constraint)?;
        let grammar = match grammar {
            Some(g) => g,
            None => {
                // If None, we probably shouldn't be creating a GuidanceState, or we create a dummy one
                // But generally the caller guards this.
                // For now, let's error if called with None, or we can handle it.
                // Actually, let's support it if needed, but for now strict.
                anyhow::bail!("Cannot create GuidanceState from Constraint::None");
            }
        };

        let parser = factory.create_parser(grammar)?;
        let matcher = Matcher::new(Ok(parser));
        Ok(Self { matcher })
    }

    pub fn compute_mask(&mut self) -> Result<Option<SimpleVob>> {
        if self.matcher.is_stopped() {
            return Ok(None);
        }
        // compute_mask returns a standard bitmask or list of tokens
        self.matcher.compute_mask().map(Some).map_err(Into::into)
    }

    pub fn commit_token(&mut self, token: u32) -> Result<()> {
        if !self.matcher.is_stopped() {
            self.matcher.consume_token(token)?;
        }
        Ok(())
    }

    pub fn is_finished(&self) -> bool {
        self.matcher.is_stopped()
    }
}

pub type ParserFactory = LlgParserFactory;

pub fn build_llg_factory(tokenizer: Tokenizer) -> Result<Arc<ParserFactory>> {
    let env = ByteTokenizer::from_tokenizer(tokenizer)?.into_tok_env(None)?;
    let factory = ParserFactory::new_simple(&env)?;
    Ok(Arc::new(factory))
}

pub fn load_toktrie_from_path(path: impl AsRef<std::path::Path>) -> Result<TokTrie> {
    let tokenizer = ByteTokenizer::from_file(path)?;
    let env = ByteTokenizerEnv::new(tokenizer, None)?;
    Ok(env.tok_trie)
}

pub fn llg_grammar_from_constraint(constraint: &Constraint) -> Result<Option<TopLevelGrammar>> {
    let grm = match constraint {
        Constraint::Regex(regex) => TopLevelGrammar::from_regex(regex),
        Constraint::Lark(lark) => TopLevelGrammar::from_lark(lark.clone()),
        Constraint::JsonSchema(value) => TopLevelGrammar::from_json_schema(value.clone()),
        Constraint::Llguidance(value) => value.clone(),
        Constraint::None => return Ok(None),
    };
    Ok(Some(grm))
}
