// src/utils/guidance.rs tests
// E2E grammar permutation tests

#[cfg(test)]
mod guidance_tests {
    use crate::tools::schema::ToolGrammarBuilder;
    use crate::tools::ToolBuilder;
    use crate::utils::guidance::*;
    use crate::utils::special_tokens::SpecialTokens;
    use crate::utils::ThinkingGrammarBuilder;
    use llguidance::api::TopLevelGrammar;
    use std::collections::HashSet;

    #[test]
    fn test_sanitize_to_ascii() {
        let input = "hello";
        let sanitized = sanitize_to_ascii(input);
        assert_eq!(sanitized, "hello");
    }

    #[test]
    fn test_sanitize_utf8_valid() {
        let input = "hello\x00\x01world";
        let sanitized = sanitize_utf8_valid(input);
        assert_eq!(sanitized, "helloworld");
    }

    #[test]
    fn test_grammar_builder_single_alternative() {
        let grammar = GrammarBuilder::new()
            .alternative(TopLevelGrammar::from_lark("start: 'a'".to_string()))
            .build();
        assert!(grammar.grammars.len() > 0);
    }

    #[test]
    fn test_grammar_builder_multiple_alternatives() {
        let grammar = GrammarBuilder::new()
            .alternative(TopLevelGrammar::from_lark("start: 'a'".to_string()))
            .alternative(TopLevelGrammar::from_lark("start: 'b'".to_string()))
            .build();
        let lark_str = get_lark_from_top_level_grammar(&grammar);
        assert!(
            lark_str.contains("start: ( 'a' | 'b' )+"),
            "Expected direct alternation"
        );
    }

    #[test]
    fn test_grammar_builder_with_max_tokens() {
        let grammar = GrammarBuilder::new()
            .alternative(TopLevelGrammar::from_lark("start: 'test'".to_string()))
            .max_tokens(100)
            .build();
        assert_eq!(grammar.max_tokens, Some(100));
    }

    #[test]
    fn test_grammar_builder_default_text() {
        let grammar = GrammarBuilder::new().build();
        let lark_str = get_lark_from_top_level_grammar(&grammar);
        assert!(
            lark_str.contains("start: text"),
            "Expected default text pattern"
        );
    }

    #[test]
    fn test_merge_top_level_grammars_direct_alternation() {
        // Test that merge_top_level_grammars produces direct alternation without rule_N indirection
        let gram1 = TopLevelGrammar::from_lark("start: 'a'".to_string());
        let gram2 = TopLevelGrammar::from_lark("start: 'b'".to_string());
        // Use None for default separator (|)
        let result = merge_top_level_grammars(vec![gram1, gram2], None, None);

        // Get the combined Lark string
        let lark_str = get_lark_from_top_level_grammar(&result);

        // Verify that start: directly alternates 'a' | 'b' without rule_N indirection
        assert!(
            lark_str.contains("start: ( 'a' | 'b' )+"),
            "Expected direct alternation in start rule: {}",
            lark_str
        );
        // Verify that rule_N indirection is NOT present
        assert!(
            !lark_str.contains("rule_0:"),
            "Should not contain rule_0 indirection"
        );
        assert!(
            !lark_str.contains("rule_1:"),
            "Should not contain rule_1 indirection"
        );
    }

    #[test]
    fn test_merge_top_level_grammars_with_text_and_tool() {
        // Test the actual TEXT | tool_call scenario from the issue
        let lark = format!("start: TEXT\n{}", chat_text_expression(false));
        let text_gram = TopLevelGrammar::from_lark(lark);
        let tool_gram =
            TopLevelGrammar::from_lark("start: tool_call\ntool_call: \"test\"".to_string());
        // Use None for default separator (|)
        let result = merge_top_level_grammars(vec![text_gram, tool_gram], None, None);

        // Get the combined Lark string
        let lark_str = get_lark_from_top_level_grammar(&result);

        // Verify that start: directly alternates TEXT | tool_call
        assert!(
            lark_str.contains("start: ( TEXT | tool_call )+"),
            "Expected direct alternation: {}",
            lark_str
        );
        // Verify that rule_N indirection is NOT present
        assert!(
            !lark_str.contains("rule_0:"),
            "Should not contain rule_0 indirection"
        );
        assert!(
            !lark_str.contains("rule_1:"),
            "Should not contain rule_1 indirection"
        );
    }

    #[test]
    fn test_merge_top_level_grammars_with_grammar_without_start() {
        // Verify that when merging a grammar without start: line, it gets properly handled
        let gram1 = TopLevelGrammar::from_lark("start: 'a'\n'a': 'a'".to_string());
        let gram2 = TopLevelGrammar::from_lark(
            "'tool': 'call'\ntool: %json {\"type\":\"object\"}".to_string(),
        );
        // Use None for default separator (|)
        let result = merge_top_level_grammars(vec![gram1, gram2], None, None);

        // Get the combined Lark string
        let lark_str = get_lark_from_top_level_grammar(&result);

        // Should still have direct alternation at start
        assert!(
            lark_str.contains("start:"),
            "Expected start rule in merged grammar"
        );
        // The tool grammar should be properly included
        assert!(
            lark_str.contains("'tool': 'call'"),
            "Expected tool content in merged grammar"
        );
    }

    #[test]
    fn test_thinking_grammar_builder_new() {
        let builder = ThinkingGrammarBuilder::new(151657, 151658, None);
        let lark = builder.build();
        assert!(
            lark.contains("reasoning_block"),
            "Should contain reasoning_block"
        );
        assert!(lark.contains("<[151657]"), "Should contain start token ID");
        assert!(lark.contains("<[151658]"), "Should contain end token ID");
    }

    #[test]
    fn test_thinking_grammar_builder_from_string() {
        let builder = ThinkingGrammarBuilder::from_string(151657, 151658);
        let lark = builder.build();
        assert!(lark.contains("<[151657]>"), "Should contain start token ID");
        assert!(lark.contains("<[151658]>"), "Should contain end token ID");
    }

    #[test]
    fn test_thinking_grammar_builder_build_grammar() {
        let builder = ThinkingGrammarBuilder::new(151657, 151658, None);
        let grammar = builder.build_grammar();
        assert!(grammar.grammars.len() > 0, "Should have grammars");
    }

    #[test]
    fn test_tool_grammar_builder_json_single_tool() {
        let tools = vec![
            ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param(
                    "query".to_string(),
                    "string".to_string(),
                    "Search query".to_string(),
                    true,
                )
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>".to_string())
            .end_tag("</tool>".to_string())
            .start_is_special(false)
            .end_is_special(false)
            .build_json();
        assert!(grammar.grammars.len() > 0);
    }

    #[test]
    fn test_tool_grammar_builder_json_multiple_tools() {
        let tools = vec![
            ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param(
                    "query".to_string(),
                    "string".to_string(),
                    "Search query".to_string(),
                    true,
                )
                .build(),
            ToolBuilder::new("weather".to_string(), "Get weather".to_string())
                .param(
                    "city".to_string(),
                    "string".to_string(),
                    "City name".to_string(),
                    true,
                )
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>".to_string())
            .end_tag("</tool>".to_string())
            .start_is_special(false)
            .end_is_special(false)
            .build_json();
        let lark_str = get_lark_from_top_level_grammar(&grammar);
        assert!(
            lark_str.contains("obj_search:"),
            "Should contain obj_search rule"
        );
        assert!(
            lark_str.contains("obj_weather:"),
            "Should contain obj_weather rule"
        );
    }

    #[test]
    fn test_tool_grammar_builder_xml_single_tool() {
        let tools = vec![
            ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param(
                    "query".to_string(),
                    "string".to_string(),
                    "Search query".to_string(),
                    true,
                )
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>".to_string())
            .end_tag("</tool>".to_string())
            .start_is_special(false)
            .end_is_special(false)
            .build_xml();
        assert!(grammar.grammars.len() > 0);
    }

    #[test]
    fn test_tool_grammar_builder_xml_multiple_tools() {
        let tools = vec![
            ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param(
                    "query".to_string(),
                    "string".to_string(),
                    "Search query".to_string(),
                    true,
                )
                .build(),
            ToolBuilder::new("weather".to_string(), "Get weather".to_string())
                .param(
                    "city".to_string(),
                    "string".to_string(),
                    "City name".to_string(),
                    true,
                )
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>".to_string())
            .end_tag("</tool>".to_string())
            .start_is_special(false)
            .end_is_special(false)
            .build_xml();
        let lark_str = get_lark_from_top_level_grammar(&grammar);
        assert!(
            lark_str.contains("tool_content: tool_0 | tool_1"),
            "Expected tool alternation"
        );
    }

    #[test]
    fn test_tool_grammar_builder_with_token_ids() {
        let tools = vec![
            ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param(
                    "query".to_string(),
                    "string".to_string(),
                    "Search query".to_string(),
                    true,
                )
                .build(),
        ];
        let mut start_ids = HashSet::new();
        start_ids.insert(151657);
        let mut end_ids = HashSet::new();
        end_ids.insert(151658);

        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("".to_string())
            .end_tag("".to_string())
            .start_is_special(false)
            .end_is_special(false)
            .start_token_ids(Some(start_ids))
            .end_token_ids(Some(end_ids))
            .build_json();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        assert!(
            lark_str.contains("<[151657]>"),
            "Should contain start token ID"
        );
        assert!(
            lark_str.contains("<[151658]>"),
            "Should contain end token ID"
        );
    }

    #[test]
    fn test_tool_grammar_builder_special_tags() {
        let tools = vec![
            ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param(
                    "query".to_string(),
                    "string".to_string(),
                    "Search query".to_string(),
                    true,
                )
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>".to_string())
            .end_tag("</tool>".to_string())
            .start_is_special(true)
            .end_is_special(true)
            .build_json();
        let lark_str = get_lark_from_top_level_grammar(&grammar);
        assert!(
            lark_str.contains("<tool>"),
            "Should contain special start tag"
        );
        assert!(
            lark_str.contains("</tool>"),
            "Should contain special end tag"
        );
    }

    #[test]
    fn test_tool_grammar_builder_empty_tools_json() {
        let grammar = ToolGrammarBuilder::new()
            .tools(&[])
            .start_tag("<tool>".to_string())
            .end_tag("</tool>".to_string())
            .start_is_special(false)
            .end_is_special(false)
            .build_json();
        let lark_str = get_lark_from_top_level_grammar(&grammar);
        assert!(
            lark_str.contains("start: tool_call"),
            "Should have start: tool_call"
        );
        assert!(
            lark_str.contains("obj: %json"),
            "Should have obj rule with generic schema"
        );
    }

    #[test]
    fn test_tool_grammar_builder_empty_tools_xml() {
        let grammar = ToolGrammarBuilder::new()
            .tools(&[])
            .start_tag("<tool>".to_string())
            .end_tag("</tool>".to_string())
            .start_is_special(false)
            .end_is_special(false)
            .build_xml();
        let lark_str = get_lark_from_top_level_grammar(&grammar);
        assert!(
            lark_str.contains("start: tool_call"),
            "Should have start: tool_call"
        );
        assert!(
            lark_str.contains("tool_content:"),
            "Should have tool_content rule"
        );
    }

    #[test]
    fn test_tool_grammar_builder_complex_schema() {
        let tools = vec![
            ToolBuilder::new("edit_file".to_string(), "Edit a file".to_string())
                .param(
                    "file_path".to_string(),
                    "string".to_string(),
                    "Path to the file".to_string(),
                    true,
                )
                .param(
                    "old_string".to_string(),
                    "string".to_string(),
                    "String to replace".to_string(),
                    true,
                )
                .param(
                    "new_string".to_string(),
                    "string".to_string(),
                    "Replacement string".to_string(),
                    true,
                )
                .param(
                    "max_replacements".to_string(),
                    "integer".to_string(),
                    "Maximum replacements".to_string(),
                    false,
                )
                .build(),
        ];

        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>".to_string())
            .end_tag("</tool>".to_string())
            .start_is_special(false)
            .end_is_special(false)
            .build_xml();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        assert!(
            lark_str.contains("param_0_0:"),
            "Should have param_0_0 rule (file_path - required)"
        );
        assert!(
            lark_str.contains("param_0_1:"),
            "Should have param_0_1 rule (old_string - required)"
        );
        assert!(
            lark_str.contains("param_0_2:"),
            "Should have param_0_2 rule (new_string - required)"
        );
        assert!(
            lark_str.contains("param_0_3:"),
            "Should have param_0_3 rule (max_replacements - optional)"
        );
    }

    // Helper to create mock SpecialTokens with all needed token IDs
    fn mock_special_tokens() -> SpecialTokens {
        SpecialTokens::from_vec(vec![
            // EOS tokens
            crate::utils::special_tokens::SpecialToken {
                category: crate::utils::special_tokens::Category::Eos,
                id: 2,
                content: b"</eos>".to_vec(),
                source: crate::utils::special_tokens::VocabSource::Special,
                normalized: false,
            },
            crate::utils::special_tokens::SpecialToken {
                category: crate::utils::special_tokens::Category::Eos,
                id: 22,
                content: b"</eof>".to_vec(),
                source: crate::utils::special_tokens::VocabSource::Special,
                normalized: false,
            },
            // Tool tokens
            crate::utils::special_tokens::SpecialToken {
                category: crate::utils::special_tokens::Category::Tool,
                id: 151657,
                content: b"<tool_call>".to_vec(),
                source: crate::utils::special_tokens::VocabSource::Added,
                normalized: false,
            },
            crate::utils::special_tokens::SpecialToken {
                category: crate::utils::special_tokens::Category::Tool,
                id: 151658,
                content: b"</tool_call>".to_vec(),
                source: crate::utils::special_tokens::VocabSource::Added,
                normalized: false,
            },
            // Reasoning tokens
            crate::utils::special_tokens::SpecialToken {
                category: crate::utils::special_tokens::Category::Reasoning,
                id: 151660,
                content: b"<thinking>".to_vec(),
                source: crate::utils::special_tokens::VocabSource::Special,
                normalized: false,
            },
            crate::utils::special_tokens::SpecialToken {
                category: crate::utils::special_tokens::Category::Reasoning,
                id: 151661,
                content: b"</thinking>".to_vec(),
                source: crate::utils::special_tokens::VocabSource::Special,
                normalized: false,
            },
        ])
    }

    fn build_mock_tool_grammar() -> TopLevelGrammar {
        let tools = vec![ToolBuilder::new("search".to_string(), "Search".to_string())
            .param("query", "string", "Query", true)
            .build()];
        ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(false)
            .end_is_special(false)
            .build_json()
    }

    fn get_lark_string(grammar: &TopLevelGrammar) -> String {
        let larks: Vec<String> = grammar
            .grammars
            .iter()
            .filter_map(|g| g.lark_grammar.as_ref())
            .map(|s| s.clone())
            .collect();
        larks.join("\n---\n")
    }

    // Test each permutation of user inputs
    #[test]
    fn test_permutation_1_no_constraints_no_tools_no_reasoning() {
        let special_tokens = mock_special_tokens();
        let tool_gram = None;
        let constraint_grammars = Vec::new();

        let result = compose_grammars(
            constraint_grammars,
            tool_gram,
            false,
            false,
            None,
            None,
            &special_tokens,
            None,
        );

        let lark = get_lark_string(&result);
        crate::log_info!("Perm 1 lark: {}", lark);
        assert!(
            lark.contains("start: text eos?"),
            "Should have text with eos?"
        );
        assert!(lark.contains("eos:"), "Should have EOS token");
        assert!(!lark.contains("tool"), "Should NOT contain tool");
    }

    #[test]
    fn test_permutation_2_no_constraints_tools_optional_no_reasoning() {
        let special_tokens = mock_special_tokens();
        let tool_gram = Some(build_mock_tool_grammar());
        let constraint_grammars = Vec::new();

        let result = compose_grammars(
            constraint_grammars,
            tool_gram,
            true,
            false,
            None,
            None,
            &special_tokens,
            None,
        );

        let lark = get_lark_string(&result);
        assert!(
            lark.contains("start: ( text | tool_call )+"),
            "Should have alternation"
        );
        assert!(lark.contains("obj_search"), "Should have tool schema");
    }

    #[test]
    fn test_permutation_3_no_constraints_tools_required_no_reasoning() {
        let special_tokens = mock_special_tokens();
        let tool_gram = Some(build_mock_tool_grammar());
        let constraint_grammars = Vec::new();

        let result = compose_grammars(
            constraint_grammars,
            tool_gram,
            true,
            true,
            None,
            None,
            &special_tokens,
            None,
        );

        let lark = get_lark_string(&result);
        assert!(
            lark.contains("start: tool_call"),
            "Should have tool_call only"
        );
        assert!(
            !lark.contains("text_with_eos"),
            "Should NOT have text_with_eos"
        );
    }

    #[test]
    fn test_permutation_4_constraint_only_no_tools_no_reasoning() {
        let special_tokens = mock_special_tokens();
        let tool_gram = None;
        let constraint_grammars = vec![TopLevelGrammarExt::from_lark_utf8(
            "start: 'option1' | 'option2'",
        )];

        let result = compose_grammars(
            constraint_grammars,
            tool_gram,
            false,
            false,
            None,
            None,
            &special_tokens,
            None,
        );

        let lark = get_lark_string(&result);
        assert!(lark.contains("option1"), "Should have constraint content");
        assert!(!lark.contains("tool"), "Should NOT have tool");
    }

    #[test]
    fn test_permutation_5_constraint_and_tools_optional_no_reasoning() {
        let special_tokens = mock_special_tokens();
        let tool_gram = Some(build_mock_tool_grammar());
        let constraint_grammars = vec![TopLevelGrammarExt::from_lark_utf8(
            "start: 'option1' | 'option2'",
        )];

        let result = compose_grammars(
            constraint_grammars,
            tool_gram,
            true,
            false,
            None,
            None,
            &special_tokens,
            None,
        );

        let lark = get_lark_string(&result);
        assert!(lark.contains("option1"), "Should have constraint content");
        assert!(lark.contains("tool_call"), "Should have tool_call");
    }

    #[test]
    fn test_permutation_6_constraint_tools_required_no_reasoning() {
        let special_tokens = mock_special_tokens();
        let tool_gram = Some(build_mock_tool_grammar());
        let constraint_grammars = vec![TopLevelGrammarExt::from_lark_utf8("start: 'option1'")];

        let result = compose_grammars(
            constraint_grammars,
            tool_gram,
            true,
            true,
            None,
            None,
            &special_tokens,
            None,
        );

        let lark = get_lark_string(&result);
        assert!(lark.contains("option1"), "Should have constraint content");
        assert!(lark.contains("tool_call"), "Should have tool_call");
    }

    #[test]
    fn test_permutation_7_constraint_forced_tool_no_reasoning() {
        let special_tokens = mock_special_tokens();
        let tool_gram = Some(build_mock_tool_grammar());
        let constraint_grammars = vec![TopLevelGrammarExt::from_lark_utf8("start: 'option1'")];

        let result = compose_grammars(
            constraint_grammars,
            tool_gram,
            true,
            false,
            Some("search".to_string()),
            None,
            &special_tokens,
            None,
        );

        let lark = get_lark_string(&result);
        assert!(lark.contains("option1"), "Should have constraint content");
        assert!(
            lark.contains("obj_search"),
            "Should have forced tool schema"
        );
    }

    #[test]
    fn test_permutation_8_no_constraints_no_tools_with_reasoning_none() {
        let special_tokens = mock_special_tokens();
        let tool_gram = None;
        let constraint_grammars = Vec::new();

        let result = compose_grammars(
            constraint_grammars,
            tool_gram,
            false,
            false,
            None,
            None,
            &special_tokens,
            Some(ReasoningEffort::None),
        );

        let lark = get_lark_string(&result);
        println!("Perm 8 lark: {}", lark);
        assert!(
            lark.contains("start: reasoning_block"),
            "Should have reasoning_block start"
        );
        assert!(lark.contains("text"), "Should have text rule");
        assert!(
            lark.contains("reasoning_block"),
            "Should have reasoning_block"
        );
        assert!(lark.contains("eos:"), "Should have EOS token");
    }

    #[test]
    fn test_permutation_9_no_constraints_tools_with_reasoning_low() {
        let special_tokens = mock_special_tokens();
        let tool_gram = Some(build_mock_tool_grammar());
        let constraint_grammars = Vec::new();

        let result = compose_grammars(
            constraint_grammars,
            tool_gram,
            true,
            false,
            None,
            None,
            &special_tokens,
            Some(ReasoningEffort::Low),
        );

        let lark = get_lark_string(&result);
        assert!(
            lark.contains("reasoning_block"),
            "Should have reasoning_block"
        );
        assert!(lark.contains("tool_call"), "Should have tool_call");
    }

    #[test]
    fn test_permutation_10_constraint_tools_with_reasoning_high() {
        let special_tokens = mock_special_tokens();
        let tool_gram = Some(build_mock_tool_grammar());
        let constraint_grammars = vec![TopLevelGrammarExt::from_lark_utf8("start: 'option1'")];

        let result = compose_grammars(
            constraint_grammars,
            tool_gram,
            true,
            false,
            None,
            None,
            &special_tokens,
            Some(ReasoningEffort::High),
        );

        let lark = get_lark_string(&result);
        assert!(
            lark.contains("reasoning_block"),
            "Should have reasoning_block"
        );
        assert!(lark.contains("option1"), "Should have constraint");
        assert!(lark.contains("tool_call"), "Should have tool_call");
    }

    #[test]
    fn test_permutation_11_reasoning_tokens_not_found_fallback() {
        // SpecialTokens without reasoning tokens
        let special_tokens =
            SpecialTokens::from_vec(vec![crate::utils::special_tokens::SpecialToken {
                category: crate::utils::special_tokens::Category::Eos,
                id: 2,
                content: b"</s>".to_vec(),
                source: crate::utils::special_tokens::VocabSource::Special,
                normalized: false,
            }]);
        let tool_gram = None;
        let constraint_grammars = Vec::new();

        let result = compose_grammars(
            constraint_grammars,
            tool_gram,
            false,
            false,
            None,
            None,
            &special_tokens,
            Some(ReasoningEffort::High),
        );

        let lark = get_lark_string(&result);
        println!("Perm 11 lark: {}", lark);
        assert!(
            lark.contains("start: text"),
            "Should fallback to text with EOS"
        );
        assert!(
            !lark.contains("reasoning"),
            "Should NOT have reasoning when tokens missing"
        );
    }

    #[test]
    fn test_with_reasoning_no_infinite_recursion() {
        // Test that WithReasoning doesn't cause infinite recursion
        // The old code used Box<GrammarComposers> which caused stack overflow
        // when to_grammar() recursively called inner.to_grammar()
        let special_tokens = mock_special_tokens();
        let tool_gram = Some(build_mock_tool_grammar());
        let constraint_grammars = vec![];

        let result = compose_grammars(
            constraint_grammars,
            tool_gram,
            true,
            false,
            None,
            None,
            &special_tokens,
            Some(ReasoningEffort::High),
        );

        let lark = get_lark_string(&result);
        // Verify the grammar was built successfully without stack overflow
        assert!(
            lark.contains("reasoning_block"),
            "Should have reasoning_block"
        );
        assert!(lark.contains("tool_call"), "Should have tool_call");
    }
}
