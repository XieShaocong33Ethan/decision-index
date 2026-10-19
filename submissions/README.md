# Submissions

| Model | Edition | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---|---:|---|---|---|---|
| [Ventus-GEN4-4B](https://huggingface.co/Ethan33X/Ventus-GEN4-4B) (`Ethan33X/Ventus-GEN4-4B`, 4.21B, Qwen3.5-4B architecture, BF16) | 0.3 | 19.71 (Raw: 37.83%) | [HF Dataset](https://huggingface.co/datasets/Ethan33X/ventus-gen4-decision-index/tree/main/Ventus-GEN4-4B) | `ventus` (`decision_index.engines.ventus_engine:VentusEngine`, single-pass causal option logit readout); zero CoT/sandboxes | 4× RTX 4080 SUPER, bf16 | Context limit 32,768 tokens; no truncation, 0% PFR |
| [Ventus-GEN4-2B](https://huggingface.co/Ethan33X/Ventus-GEN4-2B) (`Ethan33X/Ventus-GEN4-2B`, 1.88B, Qwen3.5-2B architecture, BF16) | 0.3 | 13.90 (Raw: 34.24%) | [HF Dataset](https://huggingface.co/datasets/Ethan33X/ventus-gen4-decision-index/tree/main/Ventus-GEN4-2B) | `ventus` (`decision_index.engines.ventus_engine:VentusEngine`, single-pass causal option logit readout); zero CoT/sandboxes | 4× RTX 4080 SUPER, bf16 | Context limit 32,768 tokens; no truncation, 0% PFR |
