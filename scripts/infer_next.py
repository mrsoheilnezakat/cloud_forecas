import os
import yaml
import torch
from PIL import Image
import torchvision.transforms as T

from src.models.vqvae import VQVAE
from src.models.transformer import GPT
from src.utils import top_k_sample, ensure_dir, abs_path

@torch.no_grad()
def main():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # ALWAYS use config paths (no environment variables)
    seq_dir = cfg["infer"]["seq_dir"]
    out_dir = cfg["infer"]["out_dir"]
    out_name = cfg["infer"]["out_name"]

    # Make absolute paths (prevents "where am I?" problems)
    seq_dir_abs = abs_path(seq_dir)
    out_dir_abs = abs_path(out_dir)
    ensure_dir(out_dir_abs)

    if not os.path.isdir(seq_dir_abs):
        raise RuntimeError(f"Sequence folder not found: {seq_dir_abs}")

    # Load last context frames
    image_size = cfg["data"]["image_size"]
    Tctx = cfg["data"]["context_frames"]

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])

    frames = sorted([f for f in os.listdir(seq_dir_abs) if f.lower().endswith(".png")])
    if len(frames) < Tctx:
        raise RuntimeError(f"Need at least {Tctx} frames in {seq_dir_abs}, found {len(frames)}")

    last = frames[-Tctx:]
    xs = []
    for f in last:
        img = Image.open(os.path.join(seq_dir_abs, f)).convert("L")
        xs.append(transform(img))  # (1,H,W)

    x = torch.stack(xs, dim=0).unsqueeze(0)  # (B=1,T,1,H,W)
    B, Tframes, C, H, W = x.shape

    # Load VQ-VAE
    vq = VQVAE(**cfg["vqvae"]).to(device)
    vq_ckpt_path = abs_path("checkpoints", "vqvae_final.pt")
    if not os.path.isfile(vq_ckpt_path):
        raise RuntimeError(f"Missing checkpoint: {vq_ckpt_path} (run train_vqvae.py first)")
    vq.load_state_dict(torch.load(vq_ckpt_path, map_location=device)["model"])
    vq.eval()

    # Load GPT
    vocab = cfg["vqvae"]["num_embeddings"]
    gpt = GPT(
        vocab_size=vocab,
        d_model=cfg["transformer"]["d_model"],
        n_layers=cfg["transformer"]["n_layers"],
        n_heads=cfg["transformer"]["n_heads"],
        dropout=cfg["transformer"]["dropout"],
        max_len=4096,
    ).to(device)

    gpt_ckpt_path = abs_path("checkpoints", "gpt_final.pt")
    if not os.path.isfile(gpt_ckpt_path):
        raise RuntimeError(f"Missing checkpoint: {gpt_ckpt_path} (run train_transformer.py first)")
    gpt.load_state_dict(torch.load(gpt_ckpt_path, map_location=device)["model"])
    gpt.eval()

    # Tokenize context
    ctx = x.view(B*Tframes, C, H, W).to(device)
    ctx_tokens_grid = vq.encode_indices(ctx)  # (B*T,Htok,Wtok)
    Htok, Wtok = ctx_tokens_grid.shape[-2], ctx_tokens_grid.shape[-1]
    S = Htok * Wtok

    ctx_tokens = ctx_tokens_grid.view(B, Tframes, S)   # (B,T,S)
    seq = ctx_tokens.reshape(B, Tframes*S).clone()     # (B,L0)

    # Sample next frame tokens (S tokens)
    temperature = cfg["sampling"]["temperature"]
    top_k = cfg["sampling"]["top_k"]

    for _ in range(S):
        logits = gpt(seq)                 # (B,L,V)
        next_logits = logits[:, -1, :]    # (B,V)
        next_idx = top_k_sample(next_logits, top_k=top_k, temperature=temperature)
        seq = torch.cat([seq, next_idx.unsqueeze(1)], dim=1)

    pred_flat = seq[:, -S:]              # (B,S)
    pred_grid = pred_flat.view(B, Htok, Wtok)

    pred_img = vq.decode_indices(pred_grid.to(device))  # (B,1,H,W)
    pred = pred_img[0].cpu().clamp(0, 1)                # (1,H,W)

    out_path = os.path.abspath(os.path.join(out_dir_abs, out_name))
    Image.fromarray((pred.squeeze(0).numpy() * 255).astype("uint8")).save(out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
