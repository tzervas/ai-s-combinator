//! Cross-validation binary: classifies operations from JSON input.
//!
//! Reads a JSON file of operation names, classifies each using the Rust
//! classifier, and outputs JSON results. Used by `scripts/rust_cross_validation.py`
//! to verify Python/Rust classification parity.
//!
//! Usage:
//!     cargo run --example cross_validate -- input.json [output.json]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::Instant;

use bwsk_core::classify::{Classifier, OpClass};

/// Serialize OpClass to string matching Python convention.
fn opclass_to_str(class: OpClass) -> &'static str {
    match class {
        OpClass::S => "S",
        OpClass::K => "K",
        OpClass::Gray => "GRAY",
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cross_validate <input.json> [output.json]");
        eprintln!(
            "  input.json: {{\"model_name\": str, \"ops\": [str], \"custom_rules\": {{str: str}}}}"
        );
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = args.get(2).map(|s| s.as_str());

    let input_str = fs::read_to_string(input_path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", input_path, e);
        std::process::exit(1);
    });

    // Parse input JSON manually (no serde dependency).
    // Expected format: {"model_name": "...", "ops": ["nn.Linear", ...], "custom_rules": {"Conv1D": "S"}}
    let input = parse_input(&input_str);

    let mut classifier = Classifier::new();

    // Apply custom rules
    for (name, class_str) in &input.custom_rules {
        let class = match class_str.as_str() {
            "S" => OpClass::S,
            "K" => OpClass::K,
            _ => OpClass::Gray,
        };
        classifier.add_rule(name, class);
    }

    // Classify all ops with timing
    let start = Instant::now();
    let ops_refs: Vec<&str> = input.ops.iter().map(|s| s.as_str()).collect();
    let report = classifier.classify_ops(&input.model_name, &ops_refs);
    let elapsed = start.elapsed();

    // Build output JSON
    let mut per_op = Vec::new();
    for result in &report.per_node {
        per_op.push(format!(
            "    {{\"op_type\": \"{}\", \"classification\": \"{}\", \"confidence\": {}}}",
            result.op_type,
            opclass_to_str(result.classification),
            result.confidence,
        ));
    }

    let output_json = format!(
        concat!(
            "{{\n",
            "  \"model_name\": \"{model}\",\n",
            "  \"total_ops\": {total},\n",
            "  \"s_count\": {s},\n",
            "  \"k_count\": {k},\n",
            "  \"gray_count\": {gray},\n",
            "  \"s_ratio\": {s_ratio:.6},\n",
            "  \"erasure_score\": {erasure:.6},\n",
            "  \"classify_time_us\": {time_us},\n",
            "  \"per_op\": [\n{ops}\n  ]\n",
            "}}\n",
        ),
        model = input.model_name,
        total = report.total_ops,
        s = report.s_count,
        k = report.k_count,
        gray = report.gray_count,
        s_ratio = report.s_ratio(),
        erasure = report.erasure_score,
        time_us = elapsed.as_micros(),
        ops = per_op.join(",\n"),
    );

    if let Some(path) = output_path {
        fs::write(path, &output_json).unwrap_or_else(|e| {
            eprintln!("Error writing {}: {}", path, e);
            std::process::exit(1);
        });
        eprintln!(
            "Classified {} ops in {:.1}µs -> {}",
            report.total_ops,
            elapsed.as_micros(),
            path,
        );
    } else {
        print!("{}", output_json);
    }
}

/// Simple JSON input parser (avoids serde dependency).
struct Input {
    model_name: String,
    ops: Vec<String>,
    custom_rules: HashMap<String, String>,
}

fn parse_input(json: &str) -> Input {
    // Extract model_name
    let model_name = extract_string(json, "model_name").unwrap_or_default();

    // Extract ops array
    let ops = extract_string_array(json, "ops");

    // Extract custom_rules object
    let custom_rules = extract_string_map(json, "custom_rules");

    Input {
        model_name,
        ops,
        custom_rules,
    }
}

/// Extract a string value for a given key from JSON.
fn extract_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let pos = json.find(&pattern)?;
    let rest = &json[pos + pattern.len()..];
    // Skip whitespace and colon
    let rest = rest.trim_start();
    let rest = rest.strip_prefix(':')?;
    let rest = rest.trim_start();
    // Extract quoted string
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Extract an array of strings for a given key from JSON.
fn extract_string_array(json: &str, key: &str) -> Vec<String> {
    let pattern = format!("\"{}\"", key);
    let Some(pos) = json.find(&pattern) else {
        return Vec::new();
    };
    let rest = &json[pos + pattern.len()..];
    let Some(bracket_start) = rest.find('[') else {
        return Vec::new();
    };
    let rest = &rest[bracket_start + 1..];
    let Some(bracket_end) = rest.find(']') else {
        return Vec::new();
    };
    let array_content = &rest[..bracket_end];

    let mut result = Vec::new();
    let mut pos = 0;
    while pos < array_content.len() {
        if let Some(start) = array_content[pos..].find('"') {
            let start = pos + start + 1;
            if let Some(end) = array_content[start..].find('"') {
                result.push(array_content[start..start + end].to_string());
                pos = start + end + 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    result
}

/// Extract a string->string map for a given key from JSON.
fn extract_string_map(json: &str, key: &str) -> HashMap<String, String> {
    let pattern = format!("\"{}\"", key);
    let Some(pos) = json.find(&pattern) else {
        return HashMap::new();
    };
    let rest = &json[pos + pattern.len()..];
    let Some(brace_start) = rest.find('{') else {
        return HashMap::new();
    };
    let rest = &rest[brace_start + 1..];
    let Some(brace_end) = rest.find('}') else {
        return HashMap::new();
    };
    let content = &rest[..brace_end];

    let mut result = HashMap::new();
    // Parse "key": "value" pairs
    let mut pos = 0;
    while pos < content.len() {
        // Find key
        let Some(k_start) = content[pos..].find('"') else {
            break;
        };
        let k_start = pos + k_start + 1;
        let Some(k_end) = content[k_start..].find('"') else {
            break;
        };
        let k = content[k_start..k_start + k_end].to_string();
        pos = k_start + k_end + 1;

        // Find value
        let Some(v_start) = content[pos..].find('"') else {
            break;
        };
        let v_start = pos + v_start + 1;
        let Some(v_end) = content[v_start..].find('"') else {
            break;
        };
        let v = content[v_start..v_start + v_end].to_string();
        pos = v_start + v_end + 1;

        result.insert(k, v);
    }
    result
}
