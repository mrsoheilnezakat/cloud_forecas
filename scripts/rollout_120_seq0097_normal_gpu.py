#!/usr/bin/env python3
"""
Normal (accurate) autoregressive rollout on GPU for one sequence.

- Takes 4 GT frames as context: 00500.jpg, 00501.jpg, 00502.jpg, 00503.jpg
- Predicts next 120 frames recursively (each predicted frame becomes part of context)
- For each step:
    * saves pred image (optional)
    * saves uncertainty map (optional)
    * computes IoU + PixelAcc against GT
    * computes token predictive entropy (uncertainty) from GPT logits
- Saves CSV and confirms file exists.

This matches the "normal" step-by-step token sampling (NOT one-pass).
"""

import os
import sys
import math
import glob
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch


def ensure_project_root_on_path():
    """
    Allows 'from src....' imports when running from scripts/.
    """
    here = Path(__file__).resolve()
    project_root = here.parents[1]  # cloud_forecas/
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = ensure_project_root_on_path()

from src.models.vqvae import VQVAE  # noqa: E402
from src.models.transformer import GPT  # noqa: E402


def read_gray_image(path: str, image_size: int) -> torch.Tensor:
    """
    Reads an image (.jpg/.png), converts to grayscale, resizes, returns tensor (1,1,H,W) float in [0,1].
    """
    img = Image.open(path).convert("L")
    if image_size is not None:
        img = img.resize((image_size, image_size), resample=Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, None, ...]  # (1,1,H,W)
    return t


def binarize(img_01: torch.Tensor, thr_255: int = 128) -> torch.Tensor:
    """
    img_01: (1,1,H,W) float in [0,1]
    returns: (1,1,H,W) uint8 {0,1}
    """
    thr = thr_255 / 255.0
    return (img_01 >= thr).to(torch.uint8)


def iou_and_acc(pred_bin: torch.Tensor, gt_bin: torch.Tensor) -> tuple[float, float]:
    """
    pred_bin, gt_bin: (1,1,H,W) uint8 {0,1}
    """
    pred = pred_bin.view(-1)
    gt = gt_bin.view(-1)
    inter = torch.sum((pred == 1) & (gt == 1)).item()
    union = torch.sum((pred == 1) | (gt == 1)).item()
    iou = inter / union if union > 0 else 0.0
    acc = torch.mean((pred == gt).float()).item()
    return float(iou), float(acc)


@torch.no_grad()
def sample_next_token_from_logits(logits: torch.Tensor, temperature: float, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    logits: (1, V) float
    Returns:
        next_token: (1,1) long
        entropy:    (1,1) float  (entropy of the categorical distribution)
    """
    if temperature <= 0:
        temperature = 1.0
    logits = logits / float(temperature)

    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        v, _ = torch.topk(logits, k=top_k, dim=-1)
        kth = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < kth, torch.full_like(logits, -1e9), logits)

    probs = torch.softmax(logits, dim=-1)  # (1,V)
    # entropy = -sum p log p
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1, keepdim=True)  # (1,1)

    next_token = torch.multinomial(probs, num_samples=1)  # (1,1)
    return next_token, entropy


@torch.no_grad()
def predict_next_frame_tokens_normal(
    gpt: torch.nn.Module,
    ctx_tokens_flat: torch.Tensor,
    S: int,
    block_size: int,
    temperature: float,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normal autoregressive prediction for ONE next frame:

    ctx_tokens_flat: (1, T*S) long
    Returns:
        new_tokens_flat: (1, S) long
        entropies:       (S,) float (entropy per generated token position)
    """
    device = ctx_tokens_flat.device
    x = ctx_tokens_flat.clone()  # (1, Lctx)

    ent_list = []

    for _ in range(S):
        # Keep only last block_size tokens to match positional embedding range
        x_in = x[:, -block_size:] if x.shape[1] > block_size else x
        logits_seq = gpt(x_in)  # expected (1, L, V)
        logits_last = logits_seq[:, -1, :]  # (1, V)

        nxt, ent = sample_next_token_from_logits(logits_last, temperature=temperature, top_k=top_k)
        ent_list.append(ent.squeeze().detach().float().cpu())

        x = torch.cat([x, nxt.to(device)], dim=1)

    new_flat = x[:, -S:]  # (1,S)
    entropies = torch.stack(ent_list)  # (S,)
    return new_flat, entropies


def save_gray_png(img_01: torch.Tensor, out_path: str):
    """
    img_01: (1,1,H,W) float [0,1]
    """
    arr = (img_01.squeeze().detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(out_path)


def save_uncertainty_map(entropies: torch.Tensor, token_hw: tuple[int, int], out_path: str, image_size: int):
    """
    entropies: (S,) float (CPU)
    token_hw: (Ht, Wt), Ht*Wt=S
    Creates a simple per-token entropy map and upsamples to image_size for visualization.
    """
    Ht, Wt = token_hw
    m = entropies.view(Ht, Wt).numpy().astype(np.float32)

    # Normalize to [0,1] for visualization
    mn = float(np.min(m))
    mx = float(np.max(m))
    if mx > mn:
        m = (m - mn) / (mx - mn)
    else:
        m = np.zeros_like(m)

    # Upsample nearest to image_size
    scale_h = image_size // Ht
    scale_w = image_size // Wt
    up = np.kron(m, np.ones((scale_h, scale_w), dtype=np.float32))
    up = up[:image_size, :image_size]

    arr = (up * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(out_path)


def infer_block_size_from_ckpt(gpt_state: dict) -> int:
    # pos_emb.weight is the most reliable
    if "pos_emb.weight" in gpt_state:
        return int(gpt_state["pos_emb.weight"].shape[0])
    # fallback: max_len default used in your code (but try not to hit this)
    return 4096


def infer_vocab_and_dmodel(gpt_state: dict) -> tuple[int, int]:
    # tok_emb.weight: (V, d_model)
    if "tok_emb.weight" not in gpt_state:
        raise KeyError("Expected tok_emb.weight in GPT checkpoint state_dict.")
    V = int(gpt_state["tok_emb.weight"].shape[0])
    d_model = int(gpt_state["tok_emb.weight"].shape[1])
    return V, d_model


def infer_n_layers(gpt_state: dict) -> int:
    # keys like tr.layers.0..., tr.layers.1...
    idx = set()
    for k in gpt_state.keys():
        if k.startswith("tr.layers."):
            parts = k.split(".")
            if len(parts) > 2:
                try:
                    idx.add(int(parts[2]))
                except Exception:
                    pass
    return (max(idx) + 1) if idx else 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True, help="e.g. data/test/seq_0097_MSK")
    ap.add_argument("--start_idx", type=int, default=500, help="context start index, e.g. 500 means 00500.jpg")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--vq_ckpt", required=True)
    ap.add_argument("--gpt_ckpt", required=True)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--thr", type=int, default=128, help="threshold for binarizing masks (0..255)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--pred_out_dir", default="outputs/test_rollout_120_images/seq_0097_MSK_120_normal")
    ap.add_argument("--print_every", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seq_dir = Path(args.seq_dir)
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"seq_dir not found: {seq_dir}")

    # Build file list: support .jpg/.png
    # We rely on numeric sorting by filename.
    files = sorted(
        glob.glob(str(seq_dir / "*.jpg")) + glob.glob(str(seq_dir / "*.png"))
    )
    if len(files) < (args.start_idx + 4 + args.steps):
        raise RuntimeError(
            f"Not enough frames in {seq_dir}. "
            f"Need at least {args.start_idx + 4 + args.steps}, found {len(files)}."
        )

    # ---- Load VQ-VAE ----
    vq_ck = torch.load(args.vq_ckpt, map_location=device)
    vq_state = vq_ck["model"] if isinstance(vq_ck, dict) and "model" in vq_ck else vq_ck

    vq = VQVAE().to(device)
    vq.load_state_dict(vq_state, strict=True)
    vq.eval()
    print("[VQ] Loaded.")

    # ---- Load GPT ----
    gpt_ck = torch.load(args.gpt_ckpt, map_location=device)
    gpt_state = gpt_ck["model"] if isinstance(gpt_ck, dict) and "model" in gpt_ck else gpt_ck

    block_size = infer_block_size_from_ckpt(gpt_state)
    V, d_model = infer_vocab_and_dmodel(gpt_state)
    n_layers = infer_n_layers(gpt_state)

    # n_heads is not identifiable from state dict; choose a standard valid divisor.
    # Your training used d_model=512; 8 heads is typical and works if divisible.
    n_heads = 8
    if d_model % n_heads != 0:
        # pick another safe divisor
        for h in [16, 4, 2, 1]:
            if d_model % h == 0:
                n_heads = h
                break

    # Your GPT signature (as detected earlier in this chat) supports max_len.
    gpt = GPT(vocab_size=V, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=0.1, max_len=block_size).to(device)
    gpt.load_state_dict(gpt_state, strict=True)
    gpt.eval()
    print(f"[GPT] Loaded | vocab={V} d_model={d_model} layers={n_layers} heads={n_heads} block_size={block_size}")

    # Determine token grid size from a single encode
    sample = read_gray_image(files[args.start_idx], args.image_size).to(device)
    idx_hw = vq.encode_indices(sample)  # expected (1,Ht,Wt)
    if idx_hw.ndim != 3:
        raise RuntimeError(f"Expected vq.encode_indices to return (B,Ht,Wt), got {tuple(idx_hw.shape)}")
    Ht, Wt = int(idx_hw.shape[1]), int(idx_hw.shape[2])
    S = Ht * Wt
    print(f"Token grid: {Ht}x{Wt} => S={S} tokens/frame | context tokens={4*S}")

    # Output dirs
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pred_out_dir = Path(args.pred_out_dir)
    if args.save_preds:
        pred_out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare initial context frames (4 GT frames)
    ctx_imgs = []
    for i in range(args.start_idx, args.start_idx + 4):
        ctx_imgs.append(read_gray_image(files[i], args.image_size))
    ctx_imgs = torch.cat(ctx_imgs, dim=0).to(device)  # (4,1,H,W)

    # Encode context to tokens
    ctx_tok = vq.encode_indices(ctx_imgs)  # (4,Ht,Wt)
    ctx_tok = ctx_tok.view(4, S).long()  # (4,S)
    # Flatten context as (1, T*S)
    cur_ctx = ctx_tok.unsqueeze(0).reshape(1, 4 * S).to(device)

    # Rollout loop
    rows = []
    print(f"Rolling out {args.steps} steps starting from GT {args.start_idx:05d}..{args.start_idx+3:05d}")

    for step in range(1, args.steps + 1):
        gt_idx = args.start_idx + 3 + step  # step=1 compares against frame start+4
        gt_path = files[gt_idx]

        new_flat, entropies = predict_next_frame_tokens_normal(
            gpt=gpt,
            ctx_tokens_flat=cur_ctx,
            S=S,
            block_size=block_size,
            temperature=args.temperature,
            top_k=args.top_k,
        )  # new_flat (1,S), entropies (S,) CPU

        # Decode to image
        new_hw = new_flat.view(1, Ht, Wt).to(device)  # (1,Ht,Wt)
        pred_img = vq.decode_indices(new_hw).clamp(0, 1)  # (1,1,H,W)

        # Load GT
        gt_img = read_gray_image(gt_path, args.image_size).to(device)  # (1,1,H,W)

        # Metrics on binarized masks
        pred_bin = binarize(pred_img, thr_255=args.thr)
        gt_bin = binarize(gt_img, thr_255=args.thr)
        iou, acc = iou_and_acc(pred_bin, gt_bin)

        # Uncertainty: mean predictive entropy over tokens
        unc_entropy_mean = float(entropies.mean().item())
        unc_entropy_p90 = float(torch.quantile(entropies, 0.90).item())
        unc_entropy_max = float(entropies.max().item())

        # Save outputs
        if args.save_preds:
            mean_path = pred_out_dir / f"roll_t+{step:03d}_mean_pred.png"
            unc_path = pred_out_dir / f"roll_t+{step:03d}_uncertainty.png"
            save_gray_png(pred_img, str(mean_path))
            save_uncertainty_map(entropies, (Ht, Wt), str(unc_path), image_size=args.image_size)

        rows.append({
            "step": step,
            "gt_file": os.path.basename(gt_path),
            "iou": iou,
            "pixel_acc": acc,
            "error_1_minus_iou": 1.0 - iou,
            "unc_entropy_mean": unc_entropy_mean,
            "unc_entropy_p90": unc_entropy_p90,
            "unc_entropy_max": unc_entropy_max,
            "start_idx": args.start_idx,
            "context_files": f"{os.path.basename(files[args.start_idx])},{os.path.basename(files[args.start_idx+1])},{os.path.basename(files[args.start_idx+2])},{os.path.basename(files[args.start_idx+3])}",
            "tokens_per_frame": S,
            "token_grid": f"{Ht}x{Wt}",
            "temperature": args.temperature,
            "top_k": args.top_k,
        })

        # Update context: drop oldest frame tokens, append new tokens
        # cur_ctx = [frame2, frame3, frame4, new]
        cur_ctx = torch.cat([cur_ctx[:, S:], new_flat.to(device)], dim=1)  # still (1,4*S)

        if step == 1 or step % args.print_every == 0 or step == args.steps:
            print(f"[{step:03d}/{args.steps}] IoU={iou:.4f} Acc={acc:.4f} unc_entropy_mean={unc_entropy_mean:.4f} | GT={os.path.basename(gt_path)}")

    # Save CSV
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\n[INFO] Saved CSV:")
    print(str(out_csv.resolve()))
    print("[INFO] File exists:", out_csv.exists())
    print("[DONE]")


if __name__ == "__main__":
    main()
