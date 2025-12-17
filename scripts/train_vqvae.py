import sys
import os

# ✅ Make "src" importable on Windows when running: python scripts\train_vqvae.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Keep AMP imports, but we will DISABLE AMP to avoid NaNs
from torch.cuda.amp import autocast, GradScaler

from src.utils import set_seed, ensure_dir, abs_path
from src.dataset import SequenceFolderDataset
from src.models.vqvae import VQVAE


def main():
    cfg = yaml.safe_load(open("configs/default.yaml", "r", encoding="utf-8"))
    set_seed(int(cfg.get("seed", 1337)))

    # ✅ Device selection
    want = str(cfg.get("device", "cuda")).lower()
    device = torch.device(want if (want == "cuda" and torch.cuda.is_available()) else "cpu")

    # ✅ Data
    ds = SequenceFolderDataset(
        root_dir=cfg["data"]["train_dir"],
        image_size=int(cfg["data"]["image_size"]),
        context_frames=int(cfg["data"]["context_frames"]),
        pred_frames=int(cfg["data"]["pred_frames"]),
        stride=int(cfg["data"].get("stride", 1)),
    )

    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    # ✅ Model
    model = VQVAE(**cfg["vqvae"]).to(device)
    print("Selected device:", device)
    print("Model param device:", next(model.parameters()).device)

    # ✅ Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))

    # ✅ AMP scaler (DISABLED to prevent NaNs)
    scaler = GradScaler(enabled=False)

    # ✅ Output folders
    ensure_dir(abs_path("checkpoints"))

    # ✅ Train
    model.train()
    epochs = int(cfg["train"]["epochs_vqvae"])
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))

    for epoch in range(1, epochs + 1):
        pbar = tqdm(dl, desc=f"VQ-VAE epoch {epoch}")

        for context, target in pbar:
            # context: (B,T,1,H,W)  target: (B,K,1,H,W)
            x = torch.cat([context, target], dim=1)  # (B,T+K,1,H,W)
            B, TT, C, H, W = x.shape

            # Flatten time dimension -> (B*TT, C, H, W)
            x = x.view(B * TT, C, H, W).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # ✅ AMP is OFF here to prevent numerical issues
            with autocast(enabled=False):
                x_hat, vq_loss, _ = model(x)
                recon = torch.nn.functional.l1_loss(x_hat, x)
                loss = recon + vq_loss

            # ✅ Fail fast if NaN/Inf happens
            if not torch.isfinite(loss):
                print("\n❌ Non-finite loss detected. Stopping training.")
                print("recon:", float(recon.item()), "vq:", float(vq_loss.item()), "total:", float(loss.item()))
                return

            # Backprop (normal FP32)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            pbar.set_postfix({
                "recon": f"{recon.item():.3e}",
                "vq":    f"{vq_loss.item():.3e}",
                "total": f"{loss.item():.3e}",
            })

        # Save per-epoch checkpoint
        torch.save(
            {"model": model.state_dict(), "cfg": cfg},
            abs_path("checkpoints", f"vqvae_epoch_{epoch}.pt")
        )

    # Final checkpoint
    torch.save({"model": model.state_dict(), "cfg": cfg}, abs_path("checkpoints", "vqvae_final.pt"))
    print("Saved:", abs_path("checkpoints", "vqvae_final.pt"))


if __name__ == "__main__":
    main()