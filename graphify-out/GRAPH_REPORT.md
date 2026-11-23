# Graph Report - NLP_project  (2026-11-24)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 123 nodes · 196 edges · 10 communities
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 34 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `fffeffe6`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Final Report: Benchmarking BPE vs WordPiece on Roman Urdu
- run.py
- variant_consistency
- Continued Words Chart (BPE vs WordPiece)
- Tokenizer Training (BPE & WordPiece at 1k/4k/8k/16k)
- Intrinsic Tokenizer Metrics
- Handler
- Fertility Chart
- plot
- demo.py

## God Nodes (most connected - your core abstractions)
1. `Final Report: Benchmarking BPE vs WordPiece on Roman Urdu` - 14 edges
2. `plot()` - 11 edges
3. `variant_consistency()` - 10 edges
4. `load_data()` - 9 edges
5. `intrinsic()` - 8 edges
6. `main()` - 8 edges
7. `Roman Urdu Subword Tokenizers (project)` - 8 edges
8. `clean()` - 7 edges
9. `Handler` - 6 edges
10. `Intrinsic Evaluation` - 6 edges

## Surprising Connections (you probably didn't know these)
- `intrinsic()` --implements--> `Continued Words Metric (% words split into 2+ tokens)`  [INFERRED]
  run.py → results/continued_words.png
- `intrinsic()` --implements--> `Fertility (tokens per word)`  [INFERRED]
  run.py → results/fertility.png
- `Preprocessing (lowercase, strip URLs/mentions/emojis, dedupe, 80/20 split)` --implements--> `clean()`  [INFERRED]
  README.md → run.py
- `Cleaned Dataset (19,489 sentences, stratified 80/20, seed 42)` --shares_data_with--> `clean()`  [INFERRED]
  report.pdf → run.py
- `UCI Roman Urdu Data Set` --shares_data_with--> `load_data()`  [INFERRED]
  README.md → run.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Intrinsic tokenization metrics (Rust et al.)** — readme_unk_rate, readme_fertility, readme_continued_words, readme_compression [EXTRACTED 1.00]
- **Tokenizer fragmentation vs vocab size comparison** — results_continued_words_bpe_tokenizer, results_continued_words_wordpiece_tokenizer, results_continued_words_vocabulary_size, results_continued_words_continued_words_metric [EXTRACTED 1.00]
- **Tokenizer fertility comparison across vocab sizes** — results_fertility_bpe_tokenizer, results_fertility_wordpiece_tokenizer, results_fertility_vocabulary_size, results_fertility_fertility_metric [EXTRACTED 1.00]
- **Tokenizer comparison on Macro F1 across vocab sizes** — results_macro_f1_bpe_tokenizer, results_macro_f1_wordpiece_tokenizer, results_macro_f1_word_level_baseline, results_macro_f1_vocabulary_size [EXTRACTED 1.00]
- **BPE vs WordPiece Variant Consistency Comparison** — results_variants_bpe_series, results_variants_wordpiece_series, results_variants_mean_jaccard_metric [EXTRACTED 1.00]
- **Tokenizer evaluation pipeline** — readme_preprocessing, readme_tokenizer_training, readme_intrinsic_evaluation, readme_variant_consistency, readme_downstream_evaluation [EXTRACTED 1.00]
- **Subword alone does not fix spelling variation** — report_variant_inconsistency, readme_spelling_variation, report_spelling_normalization, report_figure_variant [INFERRED 0.85]

## Communities (10 total, 0 thin omitted)

### Community 0 - "Final Report: Benchmarking BPE vs WordPiece on Roman Urdu"
Cohesion: 0.14
Nodes (20): Byte-Pair Encoding (BPE), Devlin et al. 2019 - BERT, Key Results, Pak-Austria Fachhochschule: Institute of Applied Sciences and Technology, Research Questions RQ1-RQ4, results/ directory (metrics.csv, variants.csv, downstream.csv, plots), results.ipynb, Roman Urdu Subword Tokenizers (project) (+12 more)

### Community 1 - "run.py"
Cohesion: 0.16
Nodes (19): collections, io, matplotlib, pandas, re, Preprocessing (lowercase, strip URLs/mentions/emojis, dedupe, 80/20 split), pandas, classify() (+11 more)

### Community 2 - "variant_consistency"
Cohesion: 0.15
Nodes (15): itertools, Jaccard Similarity, Roman Urdu, Spelling Variation / Non-Standardized Orthography, Spelling-Variant Consistency (Jaccard), Figure 2: Mean variant consistency vs vocab size, Spelling Normalization (phonetic / edit-distance), Finding: Consistency falls as vocab grows (frequent variants become separate whole tokens) (+7 more)

### Community 3 - "Continued Words Chart (BPE vs WordPiece)"
Cohesion: 0.50
Nodes (5): BPE Tokenizer, Continued Words Chart (BPE vs WordPiece), Continued Words Metric (% words split into 2+ tokens), Vocabulary Size (1k-16k), WordPiece Tokenizer

### Community 4 - "Tokenizer Training (BPE & WordPiece at 1k/4k/8k/16k)"
Cohesion: 0.40
Nodes (5): Tokenizer Training (BPE & WordPiece at 1k/4k/8k/16k), Recommendation: 4k-8k vocabulary, Finding: Vocabulary size matters more than algorithm, Finding: WordPiece-4k best macro F1 (63.37), not significant, HuggingFace tokenizers

### Community 5 - "Intrinsic Tokenizer Metrics"
Cohesion: 0.22
Nodes (10): Compression (characters per token), Proportion of Words Split (Continued Words), Fertility (subword tokens per word), Intrinsic Evaluation, Rust et al. 2021 - How Good is Your Tokenizer?, test_run.py metric sanity checks, Unknown-Token ([UNK]) Rate, Finding: BPE has lower fertility than WordPiece at every vocab size (+2 more)

### Community 6 - "Handler"
Cohesion: 0.38
Nodes (3): BaseHTTPRequestHandler, analyze(), Handler

### Community 7 - "Fertility Chart"
Cohesion: 0.50
Nodes (5): BPE Tokenizer, Fertility vs Vocabulary Size Chart, Fertility (tokens per word), Vocabulary Size, WordPiece Tokenizer

### Community 8 - "plot"
Cohesion: 0.16
Nodes (14): matplotlib_pyplot, Downstream Sentiment Classification (TF-IDF + Logistic Regression), Word-Level Baseline Classifier, Figure 3: Macro F1 vs vocab size with baseline, Macro F1 / Accuracy, matplotlib, scikit-learn, BPE Tokenizer (+6 more)

### Community 9 - "demo.py"
Cohesion: 0.14
Nodes (19): best_model(), main(), Compare sentiment classifiers on the same split as run.py, for 3 classes and…, Best classifier found: word 1-2 grams and character 2-5 grams, linear SVM., The two tasks: all 3 classes, and positive vs negative only., split(), tasks(), build_best() (+11 more)

## Knowledge Gaps
- **18 isolated node(s):** `jupyter`, `Pak-Austria Fachhochschule: Institute of Applied Sciences and Technology`, `GitHub repo roman-urdu-subword-tokenizers`, `results/ directory (metrics.csv, variants.csv, downstream.csv, plots)`, `Abstract` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 35 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `plot()` connect `plot` to `run.py`, `variant_consistency`, `Continued Words Chart (BPE vs WordPiece)`, `Intrinsic Tokenizer Metrics`, `Fertility Chart`?**
  _High betweenness centrality (0.312) - this node is a cross-community bridge._
- **Why does `main()` connect `run.py` to `Final Report: Benchmarking BPE vs WordPiece on Roman Urdu`, `plot`, `variant_consistency`?**
  _High betweenness centrality (0.157) - this node is a cross-community bridge._
- **Why does `Final Report: Benchmarking BPE vs WordPiece on Roman Urdu` connect `Final Report: Benchmarking BPE vs WordPiece on Roman Urdu` to `plot`, `variant_consistency`, `Intrinsic Tokenizer Metrics`?**
  _High betweenness centrality (0.156) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `plot()` (e.g. with `Figure 1: Fertility and continued words vs vocab size` and `Figure 3: Macro F1 vs vocab size with baseline`) actually correct?**
  _`plot()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `variant_consistency()` (e.g. with `Spelling-Variant Consistency (Jaccard)` and `Variant Consistency (Mean Jaccard)`) actually correct?**
  _`variant_consistency()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `load_data()` (e.g. with `UCI Roman Urdu Data Set` and `pandas`) actually correct?**
  _`load_data()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `intrinsic()` (e.g. with `Intrinsic Evaluation` and `Continued Words Metric (% words split into 2+ tokens)`) actually correct?**
  _`intrinsic()` has 3 INFERRED edges - model-reasoned connections that need verification._