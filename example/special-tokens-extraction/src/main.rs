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

    let reasoning_matches = special.search(None, Some("tool"), None, None);
    for m in reasoning_matches {
        println!("Search Result - Category: {:?}, ID: {}, string='{}'", m.category, m.id, m.string());
        // Also show the hex representation
        let hex: Vec<String> = m.content.iter().map(|b| format!("0x{:02x}", b)).collect();
        println!("  Hex: {:?}", hex);
    }

    println!("\nSuccessfully loaded tokenizer from: {}", tokenizer_path);
    println!("Total tokens processed: {}\n", special.all_tokens().len());

    // Test Eos
    println!("--- EOS Tokens ---");
    for token in special.eos() {
        println!("EOS: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }
    println!("EOS IDs: {:?}", special.eos_ids());
    println!("EOS Strings: {:?}", special.eos_strings());
    println!();

    // Test Pad
    println!("--- PAD Tokens ---");
    for token in special.pad() {
        println!("PAD: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }
    println!();

    // Test Bos
    println!("--- BOS Tokens ---");
    for token in special.bos() {
        println!("BOS: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }
    println!();

    // Test Tool
    println!("--- TOOL Tokens ---");
    for token in special.tool() {
        println!("TOOL: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }
    println!();

    // Test Role
    println!("--- ROLE Tokens ---");
    for token in special.role() {
        println!("ROLE: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }
    println!();

    // Test Mask
    println!("--- MASK Tokens ---");
    for token in special.mask() {
        println!("MASK: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }
    println!();

    // Test Reasoning
    println!("--- REASONING Tokens ---");
    for token in special.reasoning() {
        println!("REASONING: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }
    println!();

    // Test Other (Tokens that didn't match specific rules above, e.g., <unk>)
    println!("--- OTHER Tokens ---");
    for token in special.other() {
        println!("OTHER: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }
    println!();

    // Test content_type
    println!("--- CONTENT_TYPE Tokens ---");
    for token in special.content_type() {
        println!("CONTENT_TYPE: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }
    println!();

    // Test Function
    println!("--- FUNCTION Tokens ---");
    for token in special.function() {
        println!("FUNCTION: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }
    println!();

    // Test Parameter
    println!("--- PARAMETER Tokens ---");
    for token in special.parameter() {
        println!("PARAMETER: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }
    println!();

    // Test Sep
    println!("--- SEP Tokens ---");
    for token in special.sep() {
        println!("SEP: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }
    println!();

    // Test Cls
    println!("--- CLS Tokens ---");
    for token in special.cls() {
        println!("CLS: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }
    println!();

    // Test tool start/end helpers
    println!("--- Tool Start IDs ---");
    println!("{:?}", special.tool_start_ids());
    println!("--- Tool End IDs ---");
    println!("{:?}", special.tool_end_ids());
    
    // Additional search examples
    println!("\n=== Additional Search Examples ===\n");

    // Search by category only
    println!("--- Search all EOS tokens ---");
    let eos_results = special.search(None, None, Some(vllm_rs::utils::special_tokens::Category::Eos), None);
    for token in eos_results {
        println!("  EOS: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }

    // Search by source (Added tokens)
    println!("\n--- Search Added tokens ---");
    let added_results = special.search(None, None, None, Some(vllm_rs::utils::special_tokens::VocabSource::Added));
    for token in added_results {
        println!("  Added: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }

    // Search by ID
    println!("\n--- Search by ID 2 ---");
    let id_results = special.search(Some(2), None, None, None);
    for token in id_results {
        println!("  ID 2: category={:?} source={:?} string={}", token.category, token.source, token.string());
    }

    // Search by substring and category combined
    println!("\n--- Search tokens containing 'end' in EOS category ---");
    let combined_results = special.search(None, Some("end"), Some(vllm_rs::utils::special_tokens::Category::Eos), None);
    for token in combined_results {
        println!("  Token: id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }

    // Get all special tokens (Special or Added source)
    println!("\n--- All Special/Added tokens ---");
    for token in special.all_special() {
        println!("  id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }

    // Print all tokens with full details
    println!("\n=== All Tokens (Full Details) ===");
    for token in special.all_tokens() {
        println!("id={} category={:?} source={:?} string={}", token.id, token.category, token.source, token.string());
    }
}
