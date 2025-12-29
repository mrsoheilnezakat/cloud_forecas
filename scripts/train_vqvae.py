import sys, os, argparse, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import set_seed, ensure_dir, abs_path
from src.dataset import SequenceFolderDataset
from src.models.vqvae import VQVAE


def vram():
    if not torch.cuda.is_available():
        return "cpu"
    a = torch.cuda.memory_allocated() / (1024**3)
    r = torch.cuda.memory_reserved() / (1024**3)
    m = torch.cuda.max_memory_allocated() / (1024**3)
    return f"alloc={a:.2f}G reserved={r:.2f}G max={m:.2f}G"


def save_checkpoint_cpu(model: torch.nn.Module, cfg: dict, path: str, epoch: int, opt=None):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    obj = {"model": state, "cfg": cfg, "epoch": epoch}
    if opt is not None:
        # optimizer state can be huge; save it only if you really need exact resume
        try:
            obj["opt"] = opt.state_dict()
        except Exception:
            pass
    torch.save(obj, path)
    del state, obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device, opt=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    if opt is not None and "opt" in ckpt:
        try:
            opt.load_state_dict(ckpt["opt"])
        except Exception:
            pass
    return start_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run_one_epoch", action="store_true", help="Train exactly one epoch then exit.")
    ap.add_argument("--resume", default="", help="Path to checkpoint to resume from.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg.get("seed", 1337)))

    want = str(cfg.get("device", "cuda")).lower()
    device = torch.device(want if (want == "cuda" and torch.cuda.is_available()) else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ds = SequenceFolderDataset(
        root_dir=cfg["data"]["train_dir"],
        image_size=int(cfg["data"]["image_size"]),
        context_frames=int(cfg["data"]["context_frames"]),
        pred_frames=int(cfg["data"]["pred_frames"]),
        stride=int(cfg["data"].get("stride", 1)),
    )

    # IMPORTANT: disable persistent_workers to avoid any long-lived worker state
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    model = VQVAE(**cfg["vqvae"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))

    ensure_dir(abs_path("checkpoints"))
    ensure_dir(abs_path("outputs"))

    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(model, args.resume, device, opt=opt)

    epochs = int(cfg["train"]["epochs_vqvae"])
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))

    print("Selected device:", device)
    print("Model param device:", next(model.parameters()).device)
    print("Start epoch:", start_epoch, "Target epochs:", epochs)
    print("VRAM start:", vram())

    model.train()

    # decide how many epochs to run in THIS process
    end_epoch = start_epoch if args.run_one_epoch else epochs

    for epoch in range(start_epoch, end_epoch + 1):
        pbar = tqdm(dl, desc=f"VQ-VAE epoch {epoch}")
        for context, target in pbar:
            x = torch.cat([context, target], dim=1)  # (B,T+K,1,H,W)
            B, TT, C, H, W = x.shape
            x = x.view(B * TT, C, H, W).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            x_hat, vq_loss, _ = model(x)
            recon = torch.nn.functional.l1_loss(x_hat, x)
            loss = recon + vq_loss

            if not torch.isfinite(loss):
                print("\n❌ Non-finite loss detected.")
                print("VRAM:", vram())
                print("recon:", recon, "vq:", vq_loss, "total:", loss)
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            # aggressively drop references
            del x_hat, vq_loss, recon, loss, x, context, target

            if device.type == "cuda":
                pbar.set_postfix_str(vram())

        ckpt_path = abs_path("checkpoints", f"vqvae_epoch_{epoch}.pt")
        save_checkpoint_cpu(model, cfg, ckpt_path, epoch, opt=None)  # opt=None keeps file small
        print(f"Saved (CPU): {ckpt_path} | {vram()}")

        # hard cleanup at epoch boundary
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # final save
    if not args.run_one_epoch:
        final_path = abs_path("checkpoints", "vqvae_final.pt")
        save_checkpoint_cpu(model, cfg, final_path, end_epoch, opt=None)
        print("Saved final (CPU):", final_path)
        print("VRAM end:", vram())


if __name__ == "__main__":
    main()