<p align="center">
  <img src="logo_full.png" alt="PAF-IAST logo" width="600">
</p>

<h1 align="center">Roman Urdu Subword Tokenizers</h1>

<p align="center">
  <b>Benchmarking Byte-Pair Encoding (BPE) vs. WordPiece on Non-Standardized Romanized Scripts: A Roman Urdu Case Study</b><br>
  Natural Language Processing course project, Pak-Austria Fachhochschule: Institute of Applied Sciences and Technology
</p>

---

## Overview

Roman Urdu is Urdu written in the Latin script. It is widely used on social media and in chat apps, but it has no standard spelling. The same word appears in many forms:

| Meaning | Spelling variants |
|---------|-------------------|
| what    | `kya`, `kia`, `kyaa` |
| no      | `nahi`, `nahin`, `nai` |
| is      | `hai`, `hay`, `he` |

This variation makes the vocabulary large and sparse, so word-level models see many out-of-vocabulary (OOV) words. Subword tokenizers split rare words into smaller, frequent pieces. This project compares the two most common subword algorithms, **BPE** and **WordPiece**, on Roman Urdu text.

## Research Questions

1. Which algorithm, BPE or WordPiece, gives lower OOV rate and fertility on Roman Urdu?
2. How does vocabulary size change the results for each algorithm?
3. Which algorithm splits spelling variants of the same word more consistently?
4. Do differences in tokenization quality affect downstream sentiment classification?

## Dataset

[Roman Urdu Data Set](https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set) from the UCI Machine Learning Repository. It has about 20,000 social media sentences, each labeled positive, negative or neutral.

## Method

1. **Preprocess:** lowercase, remove URLs, mentions, emojis and duplicates. Split 80% train and 20% test.
2. **Train tokenizers:** BPE and WordPiece at vocabulary sizes 1k, 4k, 8k and 16k (8 tokenizers in total).
3. **Intrinsic evaluation** on the test split:
   - Unknown-token (`[UNK]`) rate
   - Fertility (subword tokens per word)
   - Proportion of words split into two or more tokens
   - Compression (characters per token)
4. **Spelling-variant consistency:** Jaccard similarity between the token sequences of spelling variants.
5. **Downstream evaluation:** TF-IDF and Logistic Regression sentiment classifier on each tokenizer's output. Report accuracy and macro F1.

## Tech Stack

- Python 3
- [HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers)
- `pandas`, `scikit-learn`, `matplotlib`
- Jupyter Notebook (no GPU needed)

## Repository Structure

```
.
├── proposal.tex      # Project proposal (LaTeX source)
├── proposal.pdf      # Project proposal (compiled)
├── logo_full.png     # University logo
└── README.md
```

Code, data and results folders will be added as the project progresses.

## Building the Proposal

```bash
pdflatex proposal.tex
```

## Project Status

- [x] Project proposal
- [ ] Data download and preprocessing
- [ ] Tokenizer training
- [ ] Intrinsic evaluation
- [ ] Spelling-variant analysis
- [ ] Downstream classification
- [ ] Final report

## Team

| Name | Registration No. |
|------|------------------|
| Qazi Absaar Baseer Ud Din | B23F0054CS069 |
| Rania Khan | B23F0082CS073 |
| Memmona Shafiq | B23F0555CS086 |

## References

- R. Sennrich, B. Haddow, A. Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016.
- M. Schuster, K. Nakajima. *Japanese and Korean Voice Search.* ICASSP 2012.
- J. Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
- P. Rust et al. *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL 2021.
- Z. Sharf. *Roman Urdu Data Set.* UCI Machine Learning Repository, 2018.
