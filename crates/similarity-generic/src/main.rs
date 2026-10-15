use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use similarity_core::apted::compute_edit_distance_with_cutoff_full_replace;
use similarity_core::cli_file_utils::collect_files;
use similarity_core::generic_parser_config::GenericParserConfig;
use similarity_core::generic_tree_sitter_parser::GenericTreeSitterParser;
use similarity_core::tree::TreeNode;
use similarity_core::APTEDOptions;
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

// Include auto-generated language configs
include!(concat!(env!("OUT_DIR"), "/language_configs.rs"));

#[derive(Parser)]
#[command(name = "similarity-generic")]
#[command(about = "Generic code similarity analyzer using tree-sitter")]
struct Cli {
    /// Files or directories to analyze (directories are walked respecting .gitignore)
    #[arg(required_unless_present_any = ["supported", "show_config"])]
    paths: Vec<PathBuf>,

    /// Language configuration file (JSON)
    #[arg(short, long, conflicts_with_all = ["language", "supported", "show_config"])]
    config: Option<PathBuf>,

    /// Language name (if using built-in config)
    #[arg(short, long, conflicts_with_all = ["config", "supported", "show_config"])]
    language: Option<String>,

    /// Similarity threshold (0.0-1.0)
    #[arg(short, long, default_value = "0.85")]
    threshold: f64,

    /// Minimum lines for functions to be considered
    #[arg(short, long, default_value = "3")]
    min_lines: u32,

    /// Rename cost for APTED algorithm
    #[arg(short, long, default_value = "0.3")]
    rename_cost: f64,

    /// Show extracted functions
    #[arg(long)]
    show_functions: bool,

    /// Show supported languages
    #[arg(long, conflicts_with_all = ["paths", "config", "language", "show_functions", "show_config"])]
    supported: bool,

    /// Show example configuration for a language
    #[arg(long, value_name = "LANGUAGE", conflicts_with_all = ["paths", "config", "language", "show_functions", "supported"])]
    show_config: Option<String>,

    /// Enable experimental overlap detection mode
    #[arg(long = "experimental-overlap")]
    overlap: bool,

    /// Minimum window size for overlap detection (number of nodes)
    #[arg(long, default_value = "8")]
    overlap_min_window: u32,

    /// Maximum window size for overlap detection (number of nodes)
    #[arg(long, default_value = "25")]
    overlap_max_window: u32,

    /// Size tolerance for overlap detection (0.0-1.0)
    #[arg(long, default_value = "0.25")]
    overlap_size_tolerance: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle --supported option
    if cli.supported {
        println!("Supported languages for generic tree-sitter parser:");
        println!("  go         - Go language");
        println!("  java       - Java language");
        println!("  c          - C language");
        println!("  cpp        - C++ language");
        println!("  csharp     - C# language");
        println!("  ruby       - Ruby language");
        println!();
        println!("Note: For Python, TypeScript, and Rust, use the dedicated implementations:");
        println!("  similarity-py  - Optimized Python analyzer");
        println!("  similarity-ts  - Optimized TypeScript/JavaScript analyzer");
        println!("  similarity-rs  - (future) Optimized Rust analyzer");
        return Ok(());
    }

    // Handle --show-config option
    if let Some(lang) = &cli.show_config {
        let config = match lang.as_str() {
            "go" => GenericParserConfig::go(),
            "java" => GenericParserConfig::java(),
            "c" => GenericParserConfig::c(),
            "cpp" | "c++" => GenericParserConfig::cpp(),
            "csharp" | "cs" => GenericParserConfig::csharp(),
            "ruby" | "rb" => GenericParserConfig::ruby(),
            _ => {
                return Err(anyhow::anyhow!(
                    "Unknown language: {}. Use --supported to see available languages.",
                    lang
                ));
            }
        };

        let json = serde_json::to_string_pretty(&config)?;
        println!("{json}");
        return Ok(());
    }

    // Normal parsing mode
    if cli.paths.is_empty() {
        return Err(anyhow::anyhow!("Path is required"));
    }

    let config = if let Some(config_path) = &cli.config {
        GenericParserConfig::from_file(config_path)
            .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?
    } else if let Some(lang) = &cli.language {
        // First try to load from embedded configs
        if let Some(config_json) =
            LANGUAGE_CONFIGS.get(lang.as_str()).or_else(|| match lang.as_str() {
                "cpp" => LANGUAGE_CONFIGS.get("cpp"),
                "c++" => LANGUAGE_CONFIGS.get("cpp"),
                "csharp" => LANGUAGE_CONFIGS.get("csharp"),
                "cs" => LANGUAGE_CONFIGS.get("csharp"),
                "ruby" => LANGUAGE_CONFIGS.get("ruby"),
                "rb" => LANGUAGE_CONFIGS.get("ruby"),
                _ => None,
            })
        {
            serde_json::from_str(config_json)
                .map_err(|e| anyhow::anyhow!("Failed to parse embedded config: {}", e))?
        } else {
            // Fall back to hardcoded configs
            match lang.as_str() {
                "go" => GenericParserConfig::go(),
                "java" => GenericParserConfig::java(),
                "c" => GenericParserConfig::c(),
                "cpp" | "c++" => GenericParserConfig::cpp(),
                "csharp" | "cs" => GenericParserConfig::csharp(),
                "ruby" | "rb" => GenericParserConfig::ruby(),
                _ => {
                    eprintln!("Error: Language '{lang}' is not supported by similarity-generic.");
                    eprintln!("Use --supported to see available languages.");
                    if matches!(
                        lang.as_str(),
                        "python" | "py" | "rust" | "rs" | "javascript" | "js" | "typescript" | "ts"
                    ) {
                        eprintln!();
                        eprintln!("Note: For {lang}, use the dedicated implementation:");
                        match lang.as_str() {
                            "python" | "py" => eprintln!("  similarity-py"),
                            "rust" | "rs" => eprintln!("  similarity-rs (planned)"),
                            "javascript" | "js" | "typescript" | "ts" => {
                                eprintln!("  similarity-ts")
                            }
                            _ => {}
                        }
                    }
                    return Err(anyhow::anyhow!("Unsupported language"));
                }
            }
        }
    } else {
        return Err(anyhow::anyhow!("Either --config or --language must be provided"));
    };

    // Create parser based on language
    let language: tree_sitter::Language = match config.language.as_str() {
        "go" => tree_sitter_go::LANGUAGE.into(),
        "java" => tree_sitter_java::LANGUAGE.into(),
        "c" => tree_sitter_c::LANGUAGE.into(),
        "cpp" => tree_sitter_cpp::LANGUAGE.into(),
        "csharp" => tree_sitter_c_sharp::LANGUAGE.into(),
        "ruby" => tree_sitter_ruby::LANGUAGE.into(),
        _ => return Err(anyhow::anyhow!("Unsupported language: {}", config.language)),
    };

    let mut parser = GenericTreeSitterParser::new(language.clone(), config.clone())
        .map_err(|e| anyhow::anyhow!("Failed to create parser: {}", e))?;

    let extensions: &[&str] = match config.language.as_str() {
        "go" => &["go"],
        "java" => &["java"],
        "c" => &["c", "h"],
        "cpp" => &["cpp", "cc", "cxx", "hpp", "hxx", "h"],
        "csharp" => &["cs"],
        "ruby" => &["rb"],
        _ => &[],
    };
    let path_strs: Vec<String> =
        cli.paths.iter().map(|p| p.to_string_lossy().to_string()).collect();
    let files = collect_files(&path_strs, extensions)?;

    if cli.overlap {
        let path = files.first().ok_or_else(|| anyhow::anyhow!("No matching file found"))?;
        return check_overlaps(
            path.clone(),
            parser,
            cli.threshold,
            cli.overlap_min_window,
            cli.overlap_max_window,
            cli.overlap_size_tolerance,
        );
    }

    struct Func {
        file: usize,
        index: usize,
        name: String,
        start_line: u32,
        end_line: u32,
        size: usize,
        labels: Vec<(usize, u32)>,
    }

    let contents: Vec<String> =
        files.iter().map(fs::read_to_string).collect::<std::io::Result<_>>()?;
    let mut label_ids: HashMap<String, usize> = HashMap::new();
    let mut functions = Vec::new();
    let mut listed = Vec::new();
    for (file_idx, content) in contents.iter().enumerate() {
        let filename = files[file_idx].to_string_lossy();
        let extracted = parser
            .extract_functions_with_trees(content)
            .map_err(|e| anyhow::anyhow!("Failed to extract functions from {}: {}", filename, e))?;
        for (index, (def, tree)) in extracted.into_iter().enumerate() {
            if cli.show_functions {
                listed.push(format!(
                    "  {} {}:{}-{}",
                    def.name, filename, def.start_line, def.end_line
                ));
            }
            if def.end_line - def.start_line + 1 < cli.min_lines {
                continue;
            }
            let mut counts: HashMap<usize, u32> = HashMap::new();
            count_labels(&tree, &mut label_ids, &mut counts);
            let mut labels: Vec<(usize, u32)> = counts.into_iter().collect();
            labels.sort_unstable();
            functions.push(Func {
                file: file_idx,
                index,
                name: def.name,
                start_line: def.start_line,
                end_line: def.end_line,
                size: tree.get_subtree_size(),
                labels,
            });
        }
    }

    if cli.show_functions {
        println!("Found {} functions:", listed.len());
        for line in &listed {
            println!("{line}");
        }
        println!();
    }

    if functions.len() < 2 {
        return Ok(());
    }
    println!("Comparing functions for similarity...");

    let apted_options = APTEDOptions {
        rename_cost: cli.rename_cost,
        delete_cost: 1.0,
        insert_cost: 1.0,
        compare_values: true,
    };

    // Lower bound on the edit distance lets most pairs skip APTED: the size difference must be
    // inserted/deleted, and every remaining unit of (label, value) histogram difference costs at
    // least min(rename / 2, 1).
    let per_label_unit = (cli.rename_cost / 2.0).min(1.0);
    functions.sort_by_key(|f| f.size);
    let candidates: Vec<(usize, usize)> = (0..functions.len())
        .flat_map(|i| {
            let functions = &functions;
            ((i + 1)..functions.len())
                .take_while(move |&j| {
                    functions[i].size as f64 >= functions[j].size as f64 * cli.threshold
                })
                .filter(move |&j| {
                    let (a, b) = (&functions[i], &functions[j]);
                    let max_size = b.size as f64;
                    let size_diff = (b.size - a.size) as f64;
                    let l1 = label_l1(&a.labels, &b.labels) as f64;
                    let lower = size_diff + (l1 - size_diff).max(0.0) * per_label_unit;
                    1.0 - lower / max_size >= cli.threshold
                })
                .map(move |j| (i, j))
        })
        .collect();

    // TreeNode uses Rc and cannot cross threads, so each worker re-extracts the trees once.
    thread_local! {
        static TREES: std::cell::RefCell<Option<Vec<Vec<Rc<TreeNode>>>>> =
            const { std::cell::RefCell::new(None) };
    }
    let mut results: Vec<(f64, usize, usize)> = candidates
        .into_par_iter()
        .filter_map(|(i, j)| {
            TREES.with(|cell| {
                let mut cell = cell.borrow_mut();
                let trees = cell.get_or_insert_with(|| {
                    let mut parser = GenericTreeSitterParser::new(language.clone(), config.clone())
                        .expect("parser was created once already");
                    contents
                        .iter()
                        .map(|c| {
                            parser
                                .extract_functions_with_trees(c)
                                .map(|v| v.into_iter().map(|(_, t)| t).collect())
                                .unwrap_or_default()
                        })
                        .collect()
                });
                let (a, b) = (&functions[i], &functions[j]);
                let (a, b) = (&trees[a.file][a.index], &trees[b.file][b.index]);
                let max_size = a.get_subtree_size().max(b.get_subtree_size()) as f64;
                let distance = compute_edit_distance_with_cutoff_full_replace(
                    a,
                    b,
                    &apted_options,
                    max_size * (1.0 - cli.threshold),
                );
                let similarity = 1.0 - distance / max_size;
                (similarity >= cli.threshold).then_some((similarity, i, j))
            })
        })
        .collect();

    let score = |r: &(f64, usize, usize)| r.0 * (functions[r.1].size + functions[r.2].size) as f64;
    results.sort_by(|x, y| score(y).partial_cmp(&score(x)).unwrap_or(std::cmp::Ordering::Equal));
    for (similarity, i, j) in results {
        let (a, b) = (&functions[i], &functions[j]);
        println!(
            "  {} <-> {}: {:.2}% ({}:{}-{}, {}:{}-{})",
            a.name,
            b.name,
            similarity * 100.0,
            files[a.file].display(),
            a.start_line,
            a.end_line,
            files[b.file].display(),
            b.start_line,
            b.end_line
        );
    }

    Ok(())
}

fn count_labels(
    node: &TreeNode,
    ids: &mut HashMap<String, usize>,
    counts: &mut HashMap<usize, u32>,
) {
    let next = ids.len();
    let id = *ids.entry(format!("{}\0{}", node.label, node.value)).or_insert(next);
    *counts.entry(id).or_default() += 1;
    for child in &node.children {
        count_labels(child, ids, counts);
    }
}

/// L1 distance between two sorted label histograms.
fn label_l1(a: &[(usize, u32)], b: &[(usize, u32)]) -> u32 {
    let (mut i, mut j, mut total) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => {
                total += a[i].1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                total += b[j].1;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                total += a[i].1.abs_diff(b[j].1);
                i += 1;
                j += 1;
            }
        }
    }
    total + a[i..].iter().map(|x| x.1).sum::<u32>() + b[j..].iter().map(|x| x.1).sum::<u32>()
}

fn check_overlaps(
    path: PathBuf,
    mut parser: GenericTreeSitterParser,
    threshold: f64,
    min_window_size: u32,
    max_window_size: u32,
    size_tolerance: f64,
) -> anyhow::Result<()> {
    use similarity_core::{find_overlaps_across_files_generic, OverlapOptions};
    use std::collections::HashMap;

    println!("Checking for overlapping code...\n");

    // Read file content
    let content = fs::read_to_string(&path)?;
    let filename = path.to_string_lossy().to_string();

    // Create file contents map
    let mut file_contents = HashMap::new();
    file_contents.insert(filename.clone(), content.clone());

    // Set up overlap options
    let options = OverlapOptions { min_window_size, max_window_size, threshold, size_tolerance };

    // Find overlaps
    let overlaps = find_overlaps_across_files_generic(&mut parser, &file_contents, &options)
        .map_err(|e| anyhow::anyhow!("Failed to find overlaps: {}", e))?;

    if overlaps.is_empty() {
        println!("\nNo code overlaps found!");
    } else {
        println!("\nCode overlaps found:");
        println!("{}", "-".repeat(60));

        for overlap_with_files in &overlaps {
            let overlap = &overlap_with_files.overlap;

            println!(
                "\nSimilarity: {:.2}% | {} nodes | {}",
                overlap.similarity * 100.0,
                overlap.node_count,
                overlap.node_type
            );
            println!(
                "  L{}-{} in function: {}",
                overlap.source_lines.0, overlap.source_lines.1, overlap.source_function
            );
            println!(
                "  L{}-{} in function: {}",
                overlap.target_lines.0, overlap.target_lines.1, overlap.target_function
            );

            // Extract and display the overlapping code
            println!("\n\x1b[36m--- Source Code ---\x1b[0m");
            if let Ok(source_segment) =
                extract_code_lines(&content, overlap.source_lines.0, overlap.source_lines.1)
            {
                println!("{source_segment}");
            }

            println!("\n\x1b[36m--- Target Code ---\x1b[0m");
            if let Ok(target_segment) =
                extract_code_lines(&content, overlap.target_lines.0, overlap.target_lines.1)
            {
                println!("{target_segment}");
            }
        }

        println!("\nTotal overlaps found: {}", overlaps.len());
    }

    Ok(())
}

fn extract_code_lines(code: &str, start_line: u32, end_line: u32) -> Result<String, String> {
    let lines: Vec<_> = code.lines().collect();

    if start_line as usize > lines.len() || end_line as usize > lines.len() {
        return Err("Line numbers out of bounds".to_string());
    }

    let start = (start_line as usize).saturating_sub(1);
    let end = (end_line as usize).min(lines.len());

    Ok(lines[start..end].join("\n"))
}
