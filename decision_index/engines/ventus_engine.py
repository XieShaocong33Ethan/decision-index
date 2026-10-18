"""ventus_engine.py

Decision Index Engine for Ventus-2B (Ethan33X/Ventus-2B).
Single-forward-pass causal decision inference evaluating direct option logits over native chat templates,
with intra-row state prefix caching and length-grouped multi-token option batching.
"""

import torch
from decision_index.engines.base import Engine, Unsupported, text
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.cache_utils import DynamicLayer, LinearAttentionLayer

SYSTEM_PROMPT = "You are a decision engine. Read the state, then answer the question by replying with exactly one option key from the list. Do not explain."


def render_prompt(state, question):
    lines = [
        "State:",
        text(state) if state not in ("", None, {}, []) else "(empty)",
        "",
        "Question:",
        text(question.get("instructions", "")),
        "",
        "Options:",
    ]
    criteria = question["criteria"] if question["type"] == "choice" else {"false": "False", "true": "True"}
    for key, description in criteria.items():
        lines.append(f"{key}: {key if description is None else text(description)}")
    lines += ["", "Reply with exactly one option key."]
    return "\n".join(lines)


def clone_cache(src: DynamicCache, config, batch_size: int = 1) -> DynamicCache:
    dst = DynamicCache(config=config)
    for idx, src_layer in enumerate(src.layers):
        dst_layer = dst.layers[idx]
        if isinstance(src_layer, DynamicLayer):
            dst_layer.dtype = src_layer.dtype
            dst_layer.device = src_layer.device
            dst_layer.is_initialized = src_layer.is_initialized
            if src_layer.is_initialized:
                if batch_size == 1:
                    dst_layer.keys = src_layer.keys.clone()
                    dst_layer.values = src_layer.values.clone()
                else:
                    dst_layer.keys = src_layer.keys.repeat_interleave(batch_size, dim=0)
                    dst_layer.values = src_layer.values.repeat_interleave(batch_size, dim=0)
        elif isinstance(src_layer, LinearAttentionLayer):
            dst_layer.dtype = src_layer.dtype
            dst_layer.device = src_layer.device
            dst_layer.record_past = src_layer.record_past
            for s_idx in range(src_layer.number_of_states):
                dst_layer.conv_kernel_size[s_idx] = src_layer.conv_kernel_size[s_idx]
                dst_layer.is_conv_states_initialized[s_idx] = src_layer.is_conv_states_initialized[s_idx]
                dst_layer.is_recurrent_states_initialized[s_idx] = src_layer.is_recurrent_states_initialized[s_idx]
                dst_layer.has_previous_state[s_idx] = src_layer.has_previous_state[s_idx]
                if src_layer.is_conv_states_initialized[s_idx]:
                    c = src_layer.conv_states[s_idx]
                    dst_layer.conv_states[s_idx] = c.clone() if batch_size == 1 else c.repeat_interleave(batch_size, dim=0)
                if src_layer.is_recurrent_states_initialized[s_idx]:
                    r = src_layer.recurrent_states[s_idx]
                    dst_layer.recurrent_states[s_idx] = r.clone() if batch_size == 1 else r.repeat_interleave(batch_size, dim=0)
    return dst


def common_prefix_len(id_lists: list[list[int]]) -> int:
    if len(id_lists) <= 1:
        return 0
    limit = min(len(ids) - 1 for ids in id_lists)
    if limit <= 0:
        return 0
    first = id_lists[0]
    p = 0
    while p < limit and all(ids[p] == first[p] for ids in id_lists[1:]):
        p += 1
    return p


class VentusEngine(Engine):
    name = "ventus"
    latency = "Device-synchronized in-process forward wall time via direct causal option logits; excludes model loading."

    def __init__(self, model="/root/autodl-tmp/models/Ventus-2B", revision=None, device=None, dtype="bfloat16", max_tokens=32768, **options):
        super().__init__(**options)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        import os
        full_state_path = os.path.join(model, "full_model_state.pt") if os.path.isdir(model) else model
        if os.path.exists(full_state_path) and full_state_path.endswith(".pt"):
            base_model_path = "/root/autodl-tmp/models/Qwen3.5-4B-Base"
            self.tok = AutoTokenizer.from_pretrained(base_model_path, revision=revision)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path, dtype=self.dtype, attn_implementation="sdpa"
            )
            state_dict = torch.load(full_state_path, map_location="cpu")
            cleaned = {k.replace("backbone.", "").replace("causal_lm.", ""): v for k, v in state_dict.items() if not k.startswith("router_head")}
            self.model.load_state_dict(cleaned, strict=False)
            del state_dict, cleaned
            self.model = self.model.to(self.device).eval()
        else:
            self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
            self.model = AutoModelForCausalLM.from_pretrained(
                model, revision=revision, dtype=self.dtype, attn_implementation="sdpa"
            ).to(self.device).eval()
        self.model_id = model
        limit = getattr(self.model.config, "max_position_embeddings", None) or getattr(self.tok, "model_max_length", None) or 262144
        self.limit = min(limit, max_tokens) if max_tokens else limit
        self.provenance = {
            "kind": "transformers",
            "repo": model,
            "revision": revision or getattr(self.model.config, "_commit_hash", None),
            "device": self.device,
            "dtype": str(dtype),
            "policy": "Every question is scored in a single causal forward pass via direct option-letter logits using standard Hugging Face AutoModelForCausalLM; probabilities are the softmax over candidate option keys.",
        }

    def runtime(self):
        import transformers
        info = {"torch": torch.__version__, "transformers": transformers.__version__, "device": self.device}
        if "cuda" in self.device:
            info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(self.device))
        return info

    def synchronize(self):
        if "cuda" in self.device:
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def __call__(self, state, questions):
        answers, raw = {}, {}
        prepared = []
        for k, q in questions.items():
            if q["type"] not in ("choice", "noul"):
                raise Unsupported(f"Unsupported question type {q['type']}")
            keys = list(q["criteria"]) if q["type"] == "choice" else ["false", "true"]
            body = render_prompt(state, q)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": body}]
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = self.tok.encode(prompt, add_special_tokens=False)
            opt_tids = []
            for key in keys:
                tids = self.tok.encode(key, add_special_tokens=False)
                if not tids:
                    tids = self.tok.encode(" " + key, add_special_tokens=False)
                if not tids:
                    raise Unsupported(f"Option key {key!r} tokenizes to nothing")
                opt_tids.append(tids)
            if len(prompt_ids) + max(len(t) for t in opt_tids) > self.limit:
                raise Unsupported(f"Prompt of {len(prompt_ids)} tokens exceeds {self.limit}-token limit")
            prepared.append((k, q, keys, opt_tids, prompt_ids))

        p = common_prefix_len([item[4] for item in prepared])
        base_cache = None
        if len(prepared) > 1 and p >= 16:
            prefix_inp = torch.tensor([prepared[0][4][:p]], device=self.device)
            base_cache = self.model(input_ids=prefix_inp, use_cache=True, logits_to_keep=1).past_key_values
        else:
            p = 0

        for k, q, keys, opt_tids, prompt_ids in prepared:
            has_multi = any(len(t) > 1 for t in opt_tids)
            if base_cache is not None:
                q_cache_in = clone_cache(base_cache, self.model.config, batch_size=1)
                q_inp = torch.tensor([prompt_ids[p:]], device=self.device)
                out = self.model(input_ids=q_inp, past_key_values=q_cache_in, use_cache=True, logits_to_keep=1)
            else:
                q_inp = torch.tensor([prompt_ids], device=self.device)
                out = self.model(input_ids=q_inp, use_cache=has_multi, logits_to_keep=1)

            first_lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)

            if not has_multi:
                scores = [first_lp[t[0]].item() for t in opt_tids]
            else:
                q_cache = out.past_key_values
                by_len = {}
                for t in opt_tids:
                    if len(t) > 1:
                        u = tuple(t[:-1])
                        by_len.setdefault(len(u), set()).add(u)
                pref_lp = {}
                for m, u_set in by_len.items():
                    u_list = sorted(u_set)
                    for start in range(0, len(u_list), 32):
                        chunk = u_list[start : start + 32]
                        bc = clone_cache(q_cache, self.model.config, batch_size=len(chunk))
                        bo = self.model(
                            input_ids=torch.tensor(chunk, device=self.device),
                            past_key_values=bc,
                            use_cache=True,
                            logits_to_keep=m,
                        )
                        blp = torch.log_softmax(bo.logits.float(), dim=-1)
                        for b_idx, u in enumerate(chunk):
                            pref_lp[u] = blp[b_idx]
                scores = []
                for t in opt_tids:
                    s = first_lp[t[0]].item()
                    if len(t) > 1:
                        u = tuple(t[:-1])
                        lp_mat = pref_lp[u]
                        for i in range(len(u)):
                            s += lp_mat[i, t[i + 1]].item()
                    scores.append(s)

            top = max(scores)
            weights = [torch.exp(torch.tensor(s - top)).item() for s in scores]
            z = sum(weights)
            probs = {key: w / z for key, w in zip(keys, weights)}
            choice = max(keys, key=lambda x: probs[x])
            raw[k] = {"option_logprobs": dict(zip(keys, scores)), "prompt_tokens": len(prompt_ids)}
            if q["type"] == "choice":
                answers[k] = {"type": "choice", "choice": choice, "probabilities": probs}
            else:
                answers[k] = {"type": "noul", "noul": probs["true"]}

        return {"model": self.model_id, "answers": answers, "usage": {"input_tokens": sum(v["prompt_tokens"] for v in raw.values())}}, raw
