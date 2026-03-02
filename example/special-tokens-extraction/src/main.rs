use vllm_rs::utils::special_tokens::SpecialTokens;
use std::env;

fn main() {
    println!("=== Testing Tokenizer Library ===\n");

    // Path to our mock tokenizer file
    let args: Vec<String> = env::args().collect();
    let tokenizer_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "./tokenizer.json".to_string()
    };

    let special = SpecialTokens::new_from_file(&tokenizer_path);

    let reasoning_matches = special.search(None, Some("think"));
    for m in reasoning_matches {
        println!("Search Result - Category: {:?}, ID: {}, Content: {}", m.category, m.id, m.content);
    }

            println!("Successfully loaded tokenizer from: {}", tokenizer_path);
            println!("Total added tokens processed.\n");

            // Test Eos
            println!("--- EOS Tokens ---");
            for (id, s) in special.eos_tokens() {
                println!("EOS: id={} token={}", id, s);
            }
            println!("EOS IDs: {:?}", special.eos_ids());
            println!("EOS Strings: {:?}", special.eos_strings());
            println!();

            // Test Pad
            println!("--- PAD Tokens ---");
            for (id, s) in special.pad_tokens() {
                println!("PAD: id={} token={}", id, s);
            }
            println!();

            // Test Bos
            println!("--- BOS Tokens ---");
            for (id, s) in special.bos_tokens() {
                println!("BOS: id={} token={}", id, s);
            }
            println!();

            // Test Tool
            println!("--- TOOL Tokens ---");
            for (id, s) in special.tool_tokens() {
                println!("TOOL: id={} token={}", id, s);
            }
            println!();

            // Test Role
            println!("--- ROLE Tokens ---");
            for (id, s) in special.role_tokens() {
                println!("ROLE: id={} token={}", id, s);
            }
            println!();

            // Test Mask
            println!("--- MASK Tokens ---");
            for (id, s) in special.mask_tokens() {
                println!("MASK: id={} token={}", id, s);
            }
            println!();

            // Test Reasoning
            println!("--- REASONING Tokens ---");
            for (id, s) in special.reasoning_tokens() {
                println!("REASONING: id={} token={}", id, s);
            }
            println!();

            // Test Other (Tokens that didn't match specific rules above, e.g., <unk>)
            println!("--- OTHER Tokens ---");
            for (id, s) in special.other_tokens() {
                println!("OTHER: id={} token={}", id, s);
            }
}

