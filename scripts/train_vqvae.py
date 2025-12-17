import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import set_seed, ensure_dir, abs_path
from src.dataset import SequenceFolderDataset
from src.models.vqvae import VQVAE

def main():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    set_seed(cfg["seed"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    ds = SequenceFolderDataset(
        root_dir=cfg["data"]["train_dir"],
        image_size=cfg["data"]["image_size"],
        context_frames=cfg["data"]["context_frames"],
        pred_frames=cfg["data"]["pred_frames"],
        stride=cfg["data"]["stride"],
    )
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    num_workers=cfg["train"]["num_workers"], pin_memory=True)

    model = VQVAE(**cfg["vqvae"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    ensure_dir(abs_path("checkpoints"))
    model.train()

    for epoch in range(1, cfg["train"]["epochs_vqvae"] + 1):
        pbar = tqdm(dl, desc=f"VQ-VAE epoch {epoch}")
        for context, target in pbar:
            x = torch.cat([context, target], dim=1)  # (B,T+K,1,H,W)
            B, TT, C, H, W = x.shape
            x = x.view(B*TT, C, H, W).to(device)

            x_hat, vq_loss, _ = model(x)
            # recon = torch.nn.functional.binary_cross_entropy(x_hat, x)
            recon = torch.nn.functional.l1_loss(x_hat, x)
            loss = recon + vq_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()

            # pbar.set_postfix(loss=float(loss.item()))
            # pbar.set_postfix(recon=float(recon.item()), vq=float(vq_loss.item()), total=float(loss.item()))
            pbar.set_postfix({"recon": f"{recon.item():.3e}",
                              "vq":    f"{vq_loss.item():.3e}",
                              "total": f"{loss.item():.3e}",
                              })


        torch.save({"model": model.state_dict(), "cfg": cfg},
                   abs_path("checkpoints", f"vqvae_epoch_{epoch}.pt"))

    torch.save({"model": model.state_dict(), "cfg": cfg}, abs_path("checkpoints", "vqvae_final.pt"))
    print("Saved:", abs_path("checkpoints", "vqvae_final.pt"))

if __name__ == "__main__":
    main()
