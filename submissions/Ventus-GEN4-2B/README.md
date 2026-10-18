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
| Jev Decision Index 0.3 | 140,178 scoreable requests (`scores.json` & `results.jsonl.gz`) |
| Model Repository | [Ethan33X/Ventus-GEN4-2B](https://huggingface.co/Ethan33X/Ventus-GEN4-2B) |
| Inference Code | [XieShaocong33Ethan/Ventus-GEN4-2B](https://github.com/XieShaocong33Ethan/Ventus-GEN4-2B) |
| Standalone Quickstart | `quickstart.py` in repository root |

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
