"""Which part of the KV-Streams context state makes Qwen3.5 end replies early?

At a call of a recorded kve episode (an eval trace with eviction records), score
the call's reply under five context states. Every variant is one full forward
pass: the GatedDeltaNet layers run over the variant's tokens (HF only carries a
DeltaNet cache across one-token steps, so a cached prefix would silently reset
it), and the 8 attention layers get an explicit mask:

  D         full context: all tokens, causal, original positions
  A_exact   kve as built: all tokens; key j is unseen by query i once i reaches the
            position its eviction applies from (the engine's and trainer's rule)
  A_static  as A_exact, but the window never sees the evicted turns (as if they
            had always been gone)
  B_static  as A_static, with the window shifted down to close the position gap
  C         sliding window: only the system prompt and the window, causal, contiguous

A_exact vs the engine's own logprobs checks the method; A_exact vs A_static is
the eviction history, A_static vs B_static the position gap, B_static vs C the
DeltaNet state over the evicted turns.

Score: the collapse is a reply "<" + end-of-message, so for each call we report
log P("<") and log P(<|im_end|> | "<") - for malformed calls (that reply was
sampled) and for normal calls after eviction (the risk it could have been).

    python eviction_diagnostic.py traces.jsonl --shard 0 --shards 4 > out.jsonl
"""

import argparse
import json
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

LT, IM_END = 27, 248046  # "<", <|im_end|>


def load(path: str, limit: int):
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                return
            yield json.loads(line)


def calls(episode: dict, max_normal: int):
    """(kind, prompt_end, stream, protected, window_start, engine_logprobs) for the
    episode's first malformed call after eviction and a few normal ones before it."""
    nodes = episode["traces"][0]["nodes"]
    stream, events, records = [], [], []
    for n in nodes:
        ids = n["token_ids"] if isinstance(n["token_ids"], list) else json.loads(n["token_ids"])
        start = len(stream)
        ev = n.get("eviction")
        ev = eval(ev) if isinstance(ev, str) else ev  # stored as a python-literal string
        if ev:
            for s, e in ev["spans"]:
                records.append((ev["at"], s, e))
        if n.get("sampled"):
            mask = n["mask"] if isinstance(n["mask"], list) else json.loads(n["mask"])
            if True in mask:
                content = str(n["message"].get("content") or "")
                lp = n.get("logprobs")
                lp = json.loads(lp) if isinstance(lp, str) else lp
                events.append((start + mask.index(True), "<action>" not in content, content, lp))
        stream += ids
    out, normals = [], []
    for prompt_end, malformed, content, lp in events:
        dead = [(s, e) for at, s, e in records if at <= prompt_end]
        if not dead:
            continue
        protected, window_start = min(s for s, _ in dead), max(e for _, e in dead)
        covered = sum(e - s for s, e in dead)
        if covered != window_start - protected:  # evictions must be the contiguous oldest turns
            continue
        item = dict(
            prompt_end=prompt_end,
            protected=protected,
            window_start=window_start,
            content=content[:40],
            engine_logprobs=lp[:2] if lp else None,
            deaths=[(at, s, e) for at, s, e in records if at <= prompt_end + 1],
        )
        if malformed:
            out.append(("malformed", item))
            break
        normals.append(("normal", item))
    return stream, out + normals[-max_normal:] if out else []


@torch.no_grad()
def forward_last2(model, tokens, positions, mask, device):
    """log P("<") at the second-to-last position and log P(<|im_end|>) at the last, one
    full forward: DeltaNet over all `tokens`, `mask` (query x key, True = visible) on
    the attention layers, `positions` for RoPE."""
    lm, head = model.model.language_model, model.lm_head
    ids = torch.tensor([tokens], device=device)
    pos = torch.tensor([positions], device=device)
    hidden = lm.embed_tokens(ids)
    rope = lm.rotary_emb(hidden, pos[None].expand(3, -1, -1))
    attn_mask = mask[None, None]
    for i, layer in enumerate(lm.layers[: lm.config.num_hidden_layers]):
        full = lm.config.layer_types[i] == "full_attention"
        hidden = layer(hidden, position_embeddings=rope, attention_mask=attn_mask if full else None, position_ids=pos)
    hidden = lm.norm(hidden[:, -2:])
    logits = head(hidden).float().log_softmax(-1)
    return float(logits[0, 0, LT]), float(logits[0, 1, IM_END])


@torch.no_grad()
def score(model, stream, item, device, check=False):
    P, W0, E = item["protected"], item["window_start"], item["prompt_end"]
    tokens = stream[:E] + [LT]
    T = len(tokens)
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    causal = j <= i
    death = torch.full((T,), -1, dtype=torch.long, device=device)
    for at, s, e in sorted(item["deaths"], reverse=True):  # first death wins
        death[s:e] = at
    exact = causal & ((death[None, :] < 0) | (i < death[None, :]))
    static = causal & ~((i >= W0) & (j >= P) & (j < W0))
    original = list(range(T))
    closed = list(range(W0)) + list(range(P, P + T - W0))
    res = {
        "D_full": forward_last2(model, tokens, original, causal, device),
        "A_exact": forward_last2(model, tokens, original, exact, device),
        "A_static": forward_last2(model, tokens, original, static, device),
        "B_static": forward_last2(model, tokens, closed, static, device),
    }
    del exact, static, causal
    window = stream[:P] + stream[W0:E] + [LT]
    n = len(window)
    res["C_sliding"] = forward_last2(model, window, list(range(n)), torch.ones(n, n, dtype=torch.bool, device=device).tril(), device)
    if check:
        # The masked layer loop with a plain causal mask must equal the stock forward.
        out = model.model.language_model(input_ids=torch.tensor([tokens], device=device))
        logits = model.lm_head(out.last_hidden_state[:, -2:]).float().log_softmax(-1)
        res["D_stock"] = (float(logits[0, 0, LT]), float(logits[0, 1, IM_END]))
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--normal", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=40000, help="skip calls with longer contexts (dense masks)")
    args = ap.parse_args()
    device = "cuda"
    model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")  # fail early if the tokenizer is missing
    first = True
    for i, episode in enumerate(load(args.traces, args.episodes)):
        if i % args.shards != args.shard:
            continue
        stream, picks = calls(episode, args.normal)
        for kind, item in picks:
            if item["prompt_end"] > args.max_tokens:
                continue
            res = score(model, stream, item, device, check=first)
            first = False
            row = {k: v for k, v in item.items() if k != "deaths"}
            print(json.dumps({"episode": i, "kind": kind, **row, **{k: list(v) for k, v in res.items()}}), flush=True)
            print(f"ep {i} {kind} len {item['prompt_end']} gap {item['window_start'] - item['protected']} {res}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
