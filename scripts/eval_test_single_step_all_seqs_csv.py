#!/usr/bin/env python3
"""
Single-step evaluation on ALL test sequences.

For each sequence in data/test:
- take frames 00001–00004
- predict frame 00005
- compute IoU + Pixel Accuracy
- save ALL results into ONE CSV file
- optionally save predicted images

Robust VQ-VAE loading:
- infers codebook size K and embedding dim D from checkpoint (vq.emb.weight)
- passes them into VQVAE using the correct __init__ argument names via inspect.signature
"""

import os
import sys
import csv
import inspect
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import argparse


# -------------------------------------------------
# Fix Python path so `import src.*` works
# -------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.vqvae import VQVAE
from src.models.transformer import GPT


# -------------------------------------------------
# Image + metrics helpers
# -------------------------------------------------
def load_gray(path, size, device):
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, None].to(device)  # (1,1,H,W)


def save_gray(x, path):
    x = x.detach().cpu().clamp(0, 1)[0, 0].numpy()
    Image.fromarray((x * 255).astype(np.uint8)).save(path)


def binarize(x, thr=0.5):
    return (x >= thr)


def compute_metrics(pred, gt):
    p = binarize(pred)
    g = binarize(gt)

    inter = (p & g).sum().item()
    union = (p | g).sum().item()

    iou = inter / union if union > 0 else 1.0
    acc = (p == g).float().mean().item()
    return iou, acc


# -------------------------------------------------
# Sampling helpers (token autoregressive)
# -------------------------------------------------
def top_k_filter(logits, k):
    if k <= 0 or k >= logits.size(-1):
        return logits
    v, _ = torch.topk(logits, k)
    thresh = v[:, -1].unsqueeze(-1)
    return torch.where(logits < thresh, torch.full_like(logits, -1e9), logits)


@torch.no_grad()
def sample_next_tokens(gpt, context_tokens, n_new, temperature=1.0, top_k=50):
    """
    context_tokens: (1, Lctx)
    returns: (1, n_new)
    """
    x = context_tokens.clone()
    out = []
    for _ in range(n_new):
        logits = gpt(x)[:, -1] / float(temperature)          # (1, V)
        logits = top_k_filter(logits, int(top_k))
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)                    # (1, 1)
        out.append(nxt)
        x = torch.cat([x, nxt], dim=1)
    return torch.cat(out, dim=1)                             # (1, n_new)


# -------------------------------------------------
# Data layout helpers
# -------------------------------------------------
def find_frame(seq_dir: Path, idx: int):
    for ext in ("png", "jpg", "jpeg"):
        p = seq_dir / f"{idx:05d}.{ext}"
        if p.exists():
            return p
    return None


# -------------------------------------------------
# Robust VQ-VAE ctor kwargs (fix 1024 vs 512 problem)
# -------------------------------------------------
def build_vqvae_from_ckpt(vq_state: dict, device: torch.device):
    """
    Infers codebook size + embedding dim from checkpoint, then builds VQVAE
    passing the correct parameter names based on VQVAE.__init__ signature.
    """
    key = "vq.emb.weight"
    if key not in vq_state:
        raise KeyError(f"Missing '{key}' in VQ-VAE state_dict; cannot infer codebook size.")

    K, D = vq_state[key].shape  # (num_embeddings, embedding_dim)
    K = int(K)
    D = int(D)

    sig = inspect.signature(VQVAE.__init__)
    params = sig.parameters

    kwargs = {}

    # map codebook size
    for name in ("num_embeddings", "n_embed", "n_codes", "codebook_size", "K"):
        if name in params:
            kwargs[name] = K
            break

    # map embedding dim
    for name in ("embedding_dim", "embed_dim", "code_dim", "D"):
        if name in params:
            kwargs[name] = D
            break

    # If neither matched, fail clearly (otherwise you silently get default=512 again)
    if not any(k in kwargs for k in ("num_embeddings", "n_embed", "n_codes", "codebook_size", "K")):
        raise RuntimeError(
            f"Could not find a codebook-size parameter in VQVAE.__init__ signature: {sig}"
        )
    if not any(k in kwargs for k in ("embedding_dim", "embed_dim", "code_dim", "D")):
        raise RuntimeError(
            f"Could not find an embedding-dim parameter in VQVAE.__init__ signature: {sig}"
        )

    print(f"[VQ] Inferred from ckpt: codebook_size={K}, embedding_dim={D}")
    print(f"[VQ] Using constructor kwargs: {kwargs}")
    print(f"[VQ] VQVAE.__init__ signature: {sig}")

    vq = VQVAE(**kwargs).to(device)
    vq.load_state_dict(vq_state, strict=True)
    vq.eval()
    return vq


def build_gpt_from_ckpt(gpt_state: dict, device: torch.device):
    """
    Builds GPT with parameters inferred from checkpoint tensors, matching your repo’s GPT signature:
      GPT(vocab_size, d_model=..., n_layers=..., n_heads=..., dropout=..., max_len=...)
    """
    vocab_size = gpt_state["tok_emb.weight"].shape[0]
    d_model = gpt_state["tok_emb.weight"].shape[1]
    max_len = gpt_state["pos_emb.weight"].shape[0]

    # infer number of layers from checkpoint keys: tr.layers.{i}.*
    layer_ids = set()
    for k in gpt_state.keys():
        if k.startswith("tr.layers."):
            parts = k.split(".")
            # tr.layers.0....
            try:
                layer_ids.add(int(parts[2]))
            except Exception:
                pass
    n_layers = (max(layer_ids) + 1) if layer_ids else 6

    # n_heads is not directly inferable robustly; use your training default
    n_heads = 8
    dropout = 0.1

    gpt = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        max_len=max_len,
    ).to(device)

    gpt.load_state_dict(gpt_state, strict=True)
    gpt.eval()

    print(f"[GPT] vocab={vocab_size} d_model={d_model} n_layers={n_layers} n_heads={n_heads} max_len={max_len}")
    return gpt


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", required=True)
    ap.add_argument("--vq_ckpt", required=True)
    ap.add_argument("--gpt_ckpt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--image_size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pred_root = out_csv.parent / "test_single_step_preds"
    if args.save_preds:
        pred_root.mkdir(parents=True, exist_ok=True)

    # ---- Load checkpoints on CPU first (safer) ----
    vq_ck = torch.load(args.vq_ckpt, map_location="cpu")
    if "model" not in vq_ck:
        raise KeyError(f"VQ checkpoint missing 'model': {args.vq_ckpt}")
    vq_state = vq_ck["model"]

    gpt_ck = torch.load(args.gpt_ckpt, map_location="cpu")
    if "model" not in gpt_ck:
        raise KeyError(f"GPT checkpoint missing 'model': {args.gpt_ckpt}")
    gpt_state = gpt_ck["model"]

    # ---- Build models (with robust VQ codebook inference) ----
    vq = build_vqvae_from_ckpt(vq_state, device)
    gpt = build_gpt_from_ckpt(gpt_state, device)

    # ---- Evaluate all sequences ----
    test_root = Path(args.test_root)
    rows = []
    header = ["sequence", "IoU", "PixelAccuracy"]

    for seq in sorted(test_root.iterdir()):
        if not seq.is_dir():
            continue

        # load context frames 1..4
        ctx = []
        for i in range(1, 5):
            p = find_frame(seq, i)
            if p is None:
                ctx = []
                break
            ctx.append(load_gray(p, args.image_size, device))

        if not ctx:
            continue

        gt_path = find_frame(seq, 5)
        if gt_path is None:
            continue
        gt = load_gray(gt_path, args.image_size, device)

        # encode context to tokens
        tokens_list = []
        idx_shape = None
        for x in ctx:
            _, _, idx = vq(x)              # idx: (1, Ht, Wt)
            idx_shape = idx.shape
            tokens_list.append(idx.view(1, -1))

        ctx_tokens = torch.cat(tokens_list, dim=1)  # (1, 4*S)
        S = idx.numel()

        # predict next frame tokens
        new_flat = sample_next_tokens(
            gpt, ctx_tokens, n_new=S,
            temperature=args.temperature,
            top_k=args.top_k
        )  # (1, S)

        new_idx = new_flat.view(*idx_shape)         # (1, Ht, Wt)
        pred = vq.decode_indices(new_idx).clamp(0, 1)

        iou, acc = compute_metrics(pred, gt)
        rows.append([seq.name, f"{iou:.6f}", f"{acc:.6f}"])

        if args.save_preds:
            out_dir = pred_root / seq.name
            out_dir.mkdir(parents=True, exist_ok=True)
            save_gray(pred, out_dir / "00005_pred.png")

        print(f"{seq.name}: IoU={iou:.4f} Acc={acc:.4f}")

    # ---- Write CSV + confirm ----
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print("\n=== CSV SAVE CONFIRMATION ===")
    print("Path:", out_csv)
    print("Exists:", out_csv.exists())
    print("Size:", out_csv.stat().st_size if out_csv.exists() else 0)
    print("Rows:", len(rows))


if __name__ == "__main__":
    main()
