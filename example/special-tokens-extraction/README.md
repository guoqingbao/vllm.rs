# Special Tokens Extraction Tool

This example tool helps developers analyze tokenizer special tokens visually using the `SpecialTokens` module from vllm.rs.

## Purpose

When integrating new models or tokenizer configurations, it's essential to understand:
- Which token IDs correspond to special tokens (EOS, BOS, PAD, TOOL, etc.)
- How the tokenizer encodes model-specific special tokens
- Whether custom token rules are needed for new model formats

This tool extracts and displays all special tokens from a tokenizer file, making it easy to visualize and verify the token mapping.

## Usage

### Basic Usage

```bash
# Using default tokenizer.json in current directory
cargo run --example special-tokens-extraction

# Using a custom tokenizer path
cargo run --example special-tokens-extraction -- /path/to/tokenizer.json
```

### Example Output

```
=== Testing Tokenizer Library ===

Successfully loaded tokenizer from: tokenizer.json
Total added tokens processed.

--- EOS Tokens ---
EOS: id=2 token=</s>
EOS: id=128001 token=<|end_of_text|>
EOS IDs: [2, 128001]
EOS Strings: ["</s>", "<|end_of_text|>"]

--- PAD Tokens ---
PAD: id=0 token=<pad>

--- BOS Tokens ---
BOS: id=1 token=<s>

--- TOOL Tokens ---
TOOL: id=151657 token=

--- ROLE Tokens ---
ROLE: id=128007 token=_ROLE
ROLE: id=128008 token=ROLE_

--- MASK Tokens ---
MASK: id=32000 token=<mask>

--- REASONING Tokens ---
REASONING: id=1 token=<unk>

--- OTHER Tokens ---
OTHER: id=0 token=<pad>
OTHER: id=1 token=<unk>
OTHER: id=2 token=<s>
OTHER: id=3 token=</s>
```

## Token Categories

The tool classifies tokens into the following categories:

| Category | Description | Example Tokens |
|----------|-------------|----------------|
| `EOS` | End of sequence tokens | `</s>`, `<|end_of_text|>`, `<eos>` |
| `PAD` | Padding tokens | `<pad>`, `<pad_token>` |
| `BOS` | Beginning of sequence tokens | `<s>`, `<|start_of_turn|>` |
| `SEP` | Separator tokens | `<sep>`, `<|separator|>` |
| `CLS` | Classification tokens | `<cls>`, `[CLS]` |
| `MASK` | Mask tokens for masking | `<mask>`, `[MASK]` |
| `TOOL` | Tool-related tokens | `<tool>`, `<|tool|>` |
| `FUNCTION` | Function tokens | `<function>`, `<|function|>` |
| `PARAMETER` | Parameter tokens | `<parameter>`, `<|parameter|>` |
| `ROLE` | Role tokens (chat templates) | `ROLE`, `ROLE_`, `<|role|>` |
| `CONTENT_TYPE` | Content type tokens | `<content_type>`, `<|content_type|>` |
| `REASONING` | Reasoning/thinking tokens | `<!--`, `-->`, `<thinking>`, `</thinking>` |
| `OTHER` | Unmatched tokens | `<unk>`, etc. |

## Understanding SpecialTokens Rules

The `SpecialTokens` struct uses a flexible matching system based on `MatchRule`:

### MatchRule Types

```rust
pub enum MatchRule {
    Exact(String),           // Exact match: "</s>" matches "</s>"
    StartsWith(String),      // Prefix match: "<|end" matches "<|end_of_text|>"
    Contains(String),        // Substring match: "tool" matches ""
    And(Box, Box),           // Both rules must match
    Or(Box, Box),            // Either rule must match
    Not(Box),                // Rule must NOT match
}
```

### Default Rules

The `default_rules()` function in `src/utils/special_tokens.rs` defines matching rules for all categories.

## Customizing Token Rules

To add support for new token patterns:

1. Edit `src/utils/special_tokens.rs`
2. Add new rules to `default_rules()`:

```rust
// Example: Add custom thinking token
(MatchRule::Contains("custom_thinking".to_string()), Category::Reasoning),

// Example: Custom tool start token
(MatchRule::Exact("<my_tool>".to_string()), Category::Tool),
```

3. Test with the extraction tool:
```bash
cargo run --example special-tokens-extraction -- /path/to/tokenizer.json
```

## Integration with vllm.rs

The `SpecialTokens` struct is used throughout vllm.rs:

### Engine Initialization

```rust
// src/core/engine.rs:474
let special_tokens = Arc::new(SpecialTokens::new(&tokenizer));
```

### Scheduler Usage

```rust
// src/core/scheduler.rs:135
eos_token_id: special_tokens.eos_ids(),
```

### Guidance/LLG Usage

```rust
// src/utils/guidance.rs:485
pub fn chat_text_expression_with_eos(special_tokens: &SpecialTokens) -> String {
    let eos_token_ids = special_tokens.eos_ids();
    // ... build TEXT pattern with EOS tokens
}
```

## Troubleshooting

### Common Issues

1. **Empty token list**
   - Check that the tokenizer file path is correct
   - Verify the tokenizer has added tokens (some tokenizers use vocab tokens)

2. **Tokens not classified correctly**
   - Add custom rules in `src/utils/special_tokens.rs`
   - Use `search(None, Some("substring"))` to debug token matching

3. **Token ID collisions**
   - The `SpecialTokens::new()` implementation deduplicates by token ID
   - Check with `search(Some(token_id), None)` to verify uniqueness

### Debugging with the Search API

```rust
// Search by ID
let matches = special_tokens.search(Some(151657), None);
for m in matches {
    println!("ID 151657: {} -> {}", m.category, m.content);
}

// Search by substring
let matches = special_tokens.search(None, Some("tool"));
for m in matches {
    println!("Contains 'tool': {} -> {}", m.category, m.content);
}
```

## File Reference

- **Source**: `src/utils/special_tokens.rs`
- **Example**: `example/special-tokens-extraction/src/main.rs`
- **Tests**: `src/utils/special_tokens.rs` (test module at end of file)

## License

This example is part of the vllm.rs project.
