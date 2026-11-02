# Tractus-decision-4B (Ethan33X)

[Tractus-decision-4B](https://huggingface.co/Ethan33X/Tractus-decision-4B) is an open-weight single-pass causal decision foundation model fine-tuned on top of Qwen3.5-4B. Operating as a pure single-pass decision model, it evaluates normalized probability distributions directly at the terminal decision position within a single causal forward pass (~163.8 ms on GPU). It generates zero free-form text tokens, uses zero Chain-of-Thought (CoT), and invokes zero external execution sandboxes. Released under CC BY-NC 4.0.

| Attribute | Tractus-decision-4B |
|---|---:|
| Parameter Scale | 4.21B Total (3.57B Non-Embedding Backbone) |
| Architecture | Qwen3.5-4B Causal LM Backbone |
| Decision Formulation | Terminal token logit projection over candidate option keys |
| Permutation Invariance | 0% Permutation Flip Rate (0% PFR) |
| JevBench 231 Total | 186 / 231 (80.52%) |
| JevBench Hard (111) | 69 / 111 (62.16%) |
| JevBench Standard (72) | 71 / 72 (98.61%) |
| JevBench Easy (48) | 46 / 48 (95.83%) |
| Jev Decision Index 0.3 | **41.31** (Raw: 55.56%) [[HF Dataset]](https://huggingface.co/datasets/Ethan33X/tractus-decision-index-0.3/tree/main/Tractus-decision-4B) |
| Model Repository | [Ethan33X/Tractus-decision-4B](https://huggingface.co/Ethan33X/Tractus-decision-4B) |
| Standalone Quickstart | `quickstart.py` in repository root |

## Decision Index 0.3 Area Breakdown

| Area | Chance-Corrected Skill | Raw Accuracy | Coverage | Benchmark Count |
|---|---:|---:|---:|---:|
| Tools & Automation | **53.63%** | 58.81% | 100.0% | 5 |
| Language Understanding | **53.19%** | 67.67% | 100.0% | 9 |
| Retrieval & Classification | **37.97%** | 53.24% | 100.0% | 6 |
| Knowledge & Reasoning | **26.06%** | 44.01% | 100.0% | 11 |
| Arts & Human Taste | **34.19%** | 52.87% | 100.0% | 6 |
| **Overall Decision Index 0.3** | **41.31** | **55.56%** | **100.0% (2,199 / 2,199)** | **37** |

## Inference Specification

- Load directly via Hugging Face `AutoModelForCausalLM.from_pretrained("Ethan33X/Tractus-decision-4B", torch_dtype=torch.bfloat16, device_map="auto")`.
- Candidate option probabilities are evaluated in a single forward pass over candidate option keys (e.g. A, B, C, D) using native chat templates.
- Native forward pass latency is ~163.8 ms on GPU, well within the Decision Index <1,000 ms SLA requirement.

## Evaluation & Reproduction

To evaluate using the official `decision-index` reproduction toolkit:

```bash
# 1. Run evaluation with native ventus engine
decision-index run ventus \
  --engine-opt model=Ethan33X/Tractus-decision-4B \
  --edition 0.3 \
  --suite-dir suite-0.3 \
  --out runs/Tractus-decision-4B

# 2. Score results and generate Decision Index metrics
decision-index score \
  --edition 0.3 \
  --suite-dir suite-0.3 \
  --results runs/Tractus-decision-4B/results.jsonl \
  --out runs/Tractus-decision-4B
```

## Declared Limits

- Context limit is 262,144 tokens; zero truncation, 0% PFR.
