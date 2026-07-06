#!/usr/bin/env python3
"""
KL divergence helper for the Llama Dashboard.
Responsibilities:
  1. Generate finetuned model response token-by-token (streaming via stdout)
  2. Run teacher-forced forward passes through both models
  3. Compute per-token KL divergence and top-token distributions
  4. Return final KL JSON

Communication protocol (stdout, one JSON line per message):
  {"type": "ready"}                        -- sent once at startup
  {"type": "token", "token": " hello"}     -- one per generated token
  {"type": "kl", "finetuned_response": "", "mean_kl": 0.0, ...}  -- once per request
"""

import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL_ID      = sys.argv[1]
FINETUNED_MODEL_ID = sys.argv[2]
DEVICE             = "cpu"
TOP_K              = 20
MAX_NEW_TOKENS     = 512

def log(msg):
    print(f"[kl_helper] {msg}", file=sys.stderr, flush=True)

def send(obj):
    print(json.dumps(obj), flush=True)

# ---- Load models -----------------------------------------------------------

log(f"Loading tokenizer from {BASE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

log("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map=DEVICE,
)
base_model.eval()

log(f"Loading finetuned model from {FINETUNED_MODEL_ID}...")
finetuned_model = AutoModelForCausalLM.from_pretrained(
    FINETUNED_MODEL_ID,
    torch_dtype=torch.float16,
    device_map=DEVICE,
)
finetuned_model.eval()

log("Both models loaded. Ready.")
send({"type": "ready"})

# ---- Format prompt in Llama-3 chat template --------------------------------

def format_prompt(user_prompt: str) -> str:
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n"
        f"You are a helpful scientific assistant specializing in "
        f"single-cell transcriptomics and computational genomics.\n"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}\n"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

# ---- Token-by-token generation ---------------------------------------------

def generate_streaming(prompt_ids: torch.Tensor):
    """
    Autoregressive generation one token at a time.
    Yields (token_id, decoded_string) for each generated token.
    Stops at EOS or MAX_NEW_TOKENS.
    """
    input_ids = prompt_ids.clone()
    past_key_values = None
    eos_id = tokenizer.eos_token_id

    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            outputs = finetuned_model(
                input_ids=input_ids if past_key_values is None
                          else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits          = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # Greedy sampling
        next_id = logits.argmax(dim=-1)
        token_str = tokenizer.decode(
            next_id, skip_special_tokens=False
        )

        yield int(next_id), token_str

        if int(next_id) == eos_id:
            break

        input_ids = torch.cat(
            [input_ids, next_id.unsqueeze(0)], dim=-1
        )

# ---- KL divergence computation ---------------------------------------------

def compute_kl(
    prompt_ids: torch.Tensor,
    generated_ids: torch.Tensor,
) -> dict:
    """
    Teacher-forced forward pass through both models over:
      full_ids = [prompt_ids | generated_ids]
    Computes KL(base || finetuned) at each generated position.
    """
    full_ids   = torch.cat([prompt_ids, generated_ids], dim=-1)
    prompt_len = prompt_ids.shape[1]
    n_total    = full_ids.shape[1]

    with torch.no_grad():
        base_logits = base_model(full_ids).logits.float()
        ft_logits   = finetuned_model(full_ids).logits.float()

    eps = 1e-10
    
    forward_kl = []
    reverse_kl = []

    for pos in range(prompt_len - 1, n_total - 1):
        p = F.softmax(base_logits[0, pos, :], dim=-1)
        q = F.softmax(ft_logits[0,   pos, :], dim=-1)
        forward = ((p + eps) * ((p + eps) / (q + eps)).log()).sum().item()

        reverse = ((q + eps) * ((q + eps) / (p + eps)).log()).sum().item()

        forward_kl.append(float(forward))
        reverse_kl.append(float(reverse))

    mean_forward = (sum(forward_kl) / len(forward_kl)
        if forward_kl else 0.0)

    mean_reverse = (sum(reverse_kl) / len(reverse_kl)
        if reverse_kl else 0.0)

    # Top-K distributions at the last generated position
    last_pos = n_total - 2
    p_last   = F.softmax(base_logits[0, last_pos, :], dim=-1)
    q_last   = F.softmax(ft_logits[0,  last_pos, :], dim=-1)

    top_base_ids = p_last.topk(TOP_K).indices.tolist()
    top_ft_ids   = q_last.topk(TOP_K).indices.tolist()
    union_ids    = list(set(top_base_ids + top_ft_ids))

    top_base, top_ft = [], []
    for tid in union_ids:
        tok = tokenizer.decode([tid])
        top_base.append([tok, float(p_last[tid])])
        top_ft.append(  [tok, float(q_last[tid])])

    top_base.sort(key=lambda x: x[1], reverse=True)
    top_ft.sort(  key=lambda x: x[1], reverse=True)

    return {
        "mean_forward_kl": mean_forward,
        "mean_reverse_kl": mean_reverse,

        "forward_kl_per_position": forward_kl,
        "reverse_kl_per_position": reverse_kl,

        "top_tokens_base": top_base[:TOP_K],
        "top_tokens_finetuned": top_ft[:TOP_K],

        "n_tokens": len(forward_kl),
    }

# ---- Main request loop -----------------------------------------------------

log("Entering request loop.")

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        req    = json.loads(line)
        prompt = req["prompt"]

        formatted    = format_prompt(prompt)
        prompt_ids   = tokenizer(
            formatted, return_tensors="pt"
        ).input_ids.to(DEVICE)

        # Step 1 — stream tokens
        generated_ids  = []
        response_parts = []

        for token_id, token_str in generate_streaming(prompt_ids):
            # Skip special tokens in display but keep ids for KL
            clean = token_str.replace("<|eot_id|>", "").replace(
                "<|end_of_text|>", "")
            if clean:
                send({"type": "token", "token": clean})
                response_parts.append(clean)
            generated_ids.append(token_id)

        finetuned_response = "".join(response_parts)

        # Step 2 — teacher-forced KL computation
        if generated_ids:
            gen_tensor = torch.tensor(
                [generated_ids], dtype=torch.long
            ).to(DEVICE)
            kl_data = compute_kl(prompt_ids, gen_tensor)
        else:
            kl_data = {
                "mean_forward_kl": 0.0,
                "mean_reverse_kl": 0.0,
                "forward_kl_per_position": [],
                "reverse_kl_per_position": [],
                "top_tokens_base": [],
                "top_tokens_finetuned": [],
                "n_tokens": 0,
            }

        # Step 3 — send final KL message
        send({
            "type":               "kl",
            "finetuned_response": finetuned_response,
            **kl_data,
        })

    except Exception as e:
        send({"type": "error", "error": str(e)})
        log(f"Error: {e}")