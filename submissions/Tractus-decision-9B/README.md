# Tractus-decision-9B (Ethan33X)

[Tractus-decision-9B](https://huggingface.co/Ethan33X/Tractus-decision-9B) is an open-weight single-pass causal decision foundation model fine-tuned on top of Qwen3.5-9B. Operating as a pure single-pass decision model, it evaluates normalized probability distributions directly at the terminal decision position within a single causal forward pass (~234.7 ms on GPU). It generates zero free-form text tokens, uses zero Chain-of-Thought (CoT), and invokes zero external execution sandboxes. Released under CC BY-NC 4.0.

| Attribute | Tractus-decision-9B |
|---|---:|
| Parameter Scale | 9.24B Total (8.12B Non-Embedding Backbone) |
| Architecture | Qwen3.5-9B Causal LM Backbone |
| Decision Formulation | Terminal token logit projection over candidate option keys |
| Permutation Invariance | 0% Permutation Flip Rate (0% PFR) |
| JevBench 231 Total | 192 / 231 (83.12%) |
| JevBench Hard (111) | 73 / 111 (65.77%) |
| JevBench Standard (72) | 72 / 72 (100.0%) |
| JevBench Easy (48) | 47 / 48 (97.92%) |
| Jev Decision Index 0.3 (`suite-0.3-probe-2.5k`) | **46.03** (Raw: 59.32%) [[HF Dataset]](https://huggingface.co/datasets/Ethan33X/tractus-decision-index-0.3/tree/main/Tractus-decision-9B) |
| Model Repository | [Ethan33X/Tractus-decision-9B](https://huggingface.co/Ethan33X/Tractus-decision-9B) |
| Standalone Quickstart | `quickstart.py` in repository root |

## Decision Index 0.3 Area Breakdown (`suite-0.3-probe-2.5k` Stratified Probe)

| Area | Chance-Corrected Skill | Raw Accuracy | Within-Probe Coverage | Benchmark Count |
|---|---:|---:|---:|---:|
| Tools & Automation | **58.15%** | 62.03% | 100.0% (257 / 257) | 5 |
| Language Understanding | **60.40%** | 73.21% | 100.0% (391 / 391) | 9 |
| Retrieval & Classification | **37.44%** | 54.77% | 100.0% (326 / 326) | 6 |
| Knowledge & Reasoning | **31.71%** | 47.33% | 100.0% (678 / 678) | 11 |
| Arts & Human Taste | **40.95%** | 58.53% | 100.0% (280 / 280) | 6 |
| **Overall Decision Index 0.3 (`suite-0.3-probe-2.5k`)** | **46.03** | **59.32%** | **100.0% (2,172 / 2,172; `complete: false`)** | **37 indexed (+6 unindexed)** |

> **Probe Provenance**: Evaluated on `suite-0.3-probe-2.5k` (`2,199` sampled requests from the `140,178`-item `release-v3` corpus, minus `27` official `excluded-questions.json` removals = `2,172` scored items across 43 catalogs: `1,932` items across 37 indexed benchmarks + `240` items across 6 unindexed reference tasks [`RouterBench=60, SGD/SGD-X=50, MMLU=40, ARC-Easy=40, ARC-Challenge=40, SimpleBench=10`]; `complete: false`). Public index weight is `20%` of the Full leaderboard composite (`80%` private tests).

## Inference Specification

- Load directly via Hugging Face `AutoModelForCausalLM.from_pretrained("Ethan33X/Tractus-decision-9B", torch_dtype=torch.bfloat16, device_map="auto")`.
- Candidate option probabilities are evaluated in a single forward pass over candidate option keys (e.g. A, B, C, D) using native chat templates.
- Native forward pass latency is ~234.7 ms on GPU, well within the Decision Index <1,000 ms SLA requirement.

## Evaluation & Reproduction

To evaluate using the official `decision-index` reproduction toolkit:

```bash
# 1. Run evaluation with native ventus engine
decision-index run ventus \
  --engine-opt model=Ethan33X/Tractus-decision-9B \
  --edition 0.3 \
  --suite-dir suite-0.3 \
  --out runs/Tractus-decision-9B

# 2. Score results and generate Decision Index metrics
decision-index score \
  --edition 0.3 \
  --suite-dir suite-0.3 \
  --results runs/Tractus-decision-9B/results.jsonl \
  --out runs/Tractus-decision-9B
```

## Declared Limits

- Context limit is 262,144 tokens; zero truncation, 0% PFR.
