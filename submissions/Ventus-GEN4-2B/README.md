# Ventus-GEN4-2B (Ethan33X)

[Ethan33X/Ventus-GEN4-2B](https://huggingface.co/Ethan33X/Ventus-GEN4-2B) is a 1.88B-parameter open-weight single-pass causal decision foundation model fine-tuned on top of Qwen3.5-2B. Operating as a pure single-pass decision model, it processes an observation state alongside structured candidate options and evaluates mathematically normalized probability distributions directly at the terminal decision position within a single forward pass (<38.5 ms on GPU). It generates zero free-form text tokens, uses zero Chain-of-Thought, and invokes zero external execution sandboxes. Released under CC BY-NC 4.0 with express commercial rights granted for fine-tuned derivatives.

| Attribute | Ventus-GEN4-2B |
|---|---:|
| Parameter Scale | 1.88B Total Parameters (1.37B Non-Embedding Backbone) |
| Architecture | Qwen3.5-2B Causal LM Backbone |
| Decision Formulation | Terminal token logit projection over candidate option keys |
| Permutation Invariance | 0% Permutation Flip Rate (0% PFR) |
| JevBench 231 Total | 176 / 231 (76.19%) [Tied #2 Sub-2B Tier Globally] |
| JevBench Hard (111) | 58 / 111 (52.25%) |
| JevBench Standard (72) | 70 / 72 (97.22%) [All-Time Record in Sub-2B Tier] |
| Jev Decision Index 0.3 | **13.90** (Raw: 34.24%, 140,178 / 140,178, 100.0% Complete) [[HF Dataset]](https://huggingface.co/datasets/Ethan33X/ventus-gen4-decision-index/tree/main/Ventus-GEN4-2B) |
| Model Repository | [Ethan33X/Ventus-GEN4-2B](https://huggingface.co/Ethan33X/Ventus-GEN4-2B) |
| Inference Code | [XieShaocong33Ethan/Ventus-GEN4-2B](https://github.com/XieShaocong33Ethan/Ventus-GEN4-2B) |
| Standalone Quickstart | `quickstart.py` in repository root |

## Decision Index 0.3 Area Breakdown

| Area | Chance-Corrected Skill | Raw Accuracy | Coverage | Benchmark Count |
|---|---:|---:|---:|---:|
| Tools & Automation | **24.87%** | 33.25% | 100.0% | 4 |
| Language Understanding | **18.41%** | 41.77% | 100.0% | 12 |
| Knowledge & Reasoning | **11.30%** | 33.29% | 100.0% | 11 |
| Arts & Human Taste | **6.79%** | 37.64% | 100.0% | 3 |
| Retrieval & Classification | **4.98%** | 24.96% | 100.0% | 7 |
| **Overall Decision Index 0.3** | **13.90** | **34.24%** | **100.0% (140,178 / 140,178)** | **37** |

## Inference Specification

- Load directly via Hugging Face `AutoModelForCausalLM.from_pretrained("Ethan33X/Ventus-GEN4-2B", torch_dtype=torch.bfloat16, device_map="auto")`.
- Candidate option probabilities are evaluated in a single forward pass over candidate option keys (e.g. A, B, C, D) using native chat templates.
- Native forward pass latency is ~38.5 ms on GPU, well within the Decision Index <1,000 ms SLA requirement on RTX PRO 6000 Ada.

## Evaluation & Reproduction

To evaluate using the official `decision-index` reproduction toolkit:

```bash
# 1. Run evaluation with native ventus engine
decision-index run ventus \
  --engine-opt model=Ethan33X/Ventus-GEN4-2B \
  --edition 0.3 \
  --suite-dir suite-0.3 \
  --out runs/Ventus-GEN4-2B

# 2. Score results and generate Decision Index metrics
decision-index score \
  --edition 0.3 \
  --suite-dir suite-0.3 \
  --results runs/Ventus-GEN4-2B/results.jsonl \
  --out runs/Ventus-GEN4-2B
```

## Declared Limits

- Context limit is 32,768 tokens; zero truncation.
