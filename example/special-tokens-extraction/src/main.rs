use std::env;
use std::fs;
use std::path::Path;
use vllm_rs::utils::special_tokens::SpecialTokens;

fn load_chat_template_from_json(template_path: &str) -> Option<String> {
    if let Ok(content) = fs::read_to_string(template_path) {
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;
        json.get("chat_template")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    } else {
        None
    }
}

fn load_chat_template_from_jinja(template_path: &str) -> Option<String> {
    fs::read_to_string(template_path).ok()
}

fn main() {
    println!("=== Testing Tokenizer Library ===\n");

    let args: Vec<String> = env::args().collect();

    let tokenizer_path = if args.len() > 1 {
        let path = args[1].clone();
        if path.ends_with("tokenizer_config.json") {
            let parent_dir = Path::new(&path).parent().unwrap_or(Path::new("."));
            let tokenizer_json = parent_dir.join("tokenizer.json");
            if tokenizer_json.exists() {
                tokenizer_json.to_string_lossy().into_owned()
            } else {
                path
            }
        } else {
            path
        }
    } else {
        "./tokenizer.json".to_string()
    };

    let chat_template_path = args.get(2).map(|s| s.as_str());

    println!("--- Loading Tokenizer ---");
    println!("Tokenizer path: {}", tokenizer_path);

    let special = SpecialTokens::new_from_file(&tokenizer_path);

    println!("Successfully loaded tokenizer.");
    println!("Total tokens: {}\n", special.all_tokens().len());

    // Load chat template
    let template_path = if let Some(p) = chat_template_path {
        // Resolve relative paths against current directory
        let path = Path::new(p);
        if path.is_relative() {
            let current_dir = std::env::current_dir().unwrap_or(Path::new(".").to_path_buf());
            Some(current_dir.join(path).to_string_lossy().into_owned())
        } else {
            Some(p.to_string())
        }
    } else {
        let tokenizer_dir = Path::new(&tokenizer_path)
            .parent()
            .unwrap_or(Path::new("."));
        let jinja_path = tokenizer_dir.join("chat_template.jinja");
        if jinja_path.exists() {
            Some(jinja_path.to_string_lossy().into_owned())
        } else {
            None
        }
    };

    if let Some(template_path) = template_path {
        println!("--- Chat Template ---");
        println!("Template path: {}", template_path);

        let chat_template = if template_path.ends_with(".json") {
            load_chat_template_from_json(&template_path)
        } else if template_path.ends_with(".jinja") {
            load_chat_template_from_jinja(&template_path)
        } else {
            load_chat_template_from_jinja(&template_path)
                .or_else(|| load_chat_template_from_json(&template_path))
        };

        match chat_template {
            Some(template) => {
                println!("Template loaded successfully!");
                println!("Template length: {} characters", template.len());

                // Parse and display template sections
                println!("\n--- Template Structure ---\n");
                parse_template_structure(&template);
            }
            None => {
                println!("No chat template found in: {}", template_path);
            }
        }
        println!();
    }

    // Display special tokens by category
    println!("--- EOS Tokens ---");
    for token in special.eos() {
        println!(
            "  id={} category={:?} source={:?} string={}",
            token.id,
            token.category,
            token.source,
            token.string()
        );
    }
    println!();

    println!("--- TOOL Tokens ---");
    for token in special.tool() {
        println!(
            "  id={} category={:?} source={:?} string={}",
            token.id,
            token.category,
            token.source,
            token.string()
        );
    }
    println!();

    println!("--- FUNCTION Tokens ---");
    for token in special.function() {
        println!(
            "  id={} category={:?} source={:?} string={}",
            token.id,
            token.category,
            token.source,
            token.string()
        );
    }
    println!();

    println!("--- PARAMETER Tokens ---");
    for token in special.parameter() {
        println!(
            "  id={} category={:?} source={:?} string={}",
            token.id,
            token.category,
            token.source,
            token.string()
        );
    }
    println!();

    println!("--- Reasoning Tokens ---");
    for token in special.reasoning() {
        println!(
            "  id={} category={:?} source={:?} string={}",
            token.id,
            token.category,
            token.source,
            token.string()
        );
    }
    println!();

    println!("--- Tool Start/End IDs ---");
    println!("tool_start_ids: {:?}", special.tool_start_ids());
    println!("tool_end_ids: {:?}", special.tool_end_ids());
    println!();

    println!("--- Reasoning Start/End IDs ---");
    println!("reasoning_start_ids: {:?}", special.reasoning_start_ids());
    println!("reasoning_end_ids: {:?}", special.reasoning_end_ids());
    println!();

    println!("=== All Tokens ===");
    for token in special.all_special() {
        println!(
            "id={} category={:?} source={:?} string={}",
            token.id,
            token.category,
            token.source,
            token.string()
        );
    }
}

fn parse_template_structure(template: &str) {
    // Find and display key structural blocks
    let blocks = vec![
        ("Iterators", vec!["{%- set", "{%- macro", "{% set"]),
        ("Macros", vec!["{% macro", "{%- macro"]),
        ("Tools", vec!["{% if tools", "{%- if tools"]),
        (
            "System Message",
            vec!["system", "{%- if messages[0].role == 'system'"],
        ),
        (
            "User Message",
            vec![
                "{% elif message.role == \"user\"",
                "{%- elif message.role == \"user\"",
            ],
        ),
        (
            "Assistant Message",
            vec![
                "{% elif message.role == \"assistant\"",
                "{%- elif message.role == \"assistant\"",
            ],
        ),
        (
            "Tool Response",
            vec!["{% elif message.role == \"tool\"", "tool_response"],
        ),
        ("Thinking", vec!["{% think", "{{- '<‌think>"]),
        (
            "Generation Prompt",
            vec!["{% if add_generation_prompt", "{{- '<‌|im_start|>assistant"],
        ),
    ];

    for (block_name, keywords) in blocks {
        println!("[=== {} ===]", block_name);
        let found_lines: Vec<&str> = template
            .lines()
            .filter(|line| keywords.iter().any(|k| line.contains(k)))
            .collect();

        if !found_lines.is_empty() {
            for line in found_lines {
                println!("{}", line);
            }
        } else {
            println!("  (no lines found with these keywords)");
        }
        println!();
    }

    // Show full template at the end
    println!("[=== FULL TEMPLATE ({} chars) ===]", template.len());
    println!("{}", template);
}
