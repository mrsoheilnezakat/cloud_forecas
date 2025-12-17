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
from src.models.transformer import GPT

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

    # VQ-VAE (frozen)
    vq = VQVAE(**cfg["vqvae"]).to(device)
    ck = torch.load(abs_path("checkpoints", "vqvae_final.pt"), map_location=device)
    vq.load_state_dict(ck["model"])
    vq.eval()
    for p in vq.parameters():
        p.requires_grad = False

    vocab = cfg["vqvae"]["num_embeddings"]
    gpt = GPT(
        vocab_size=vocab,
        d_model=cfg["transformer"]["d_model"],
        n_layers=cfg["transformer"]["n_layers"],
        n_heads=cfg["transformer"]["n_heads"],
        dropout=cfg["transformer"]["dropout"],
        max_len=4096,
    ).to(device)

    opt = torch.optim.AdamW(gpt.parameters(), lr=cfg["train"]["lr"])
    ensure_dir(abs_path("checkpoints"))

    gpt.train()

    for epoch in range(1, cfg["train"]["epochs_transformer"] + 1):
        pbar = tqdm(dl, desc=f"GPT epoch {epoch}")
        for context, target in pbar:
            B, T, C, H, W = context.shape
            K = target.shape[1]

            ctx = context.view(B*T, C, H, W).to(device)
            tgt = target.view(B*K, C, H, W).to(device)

            with torch.no_grad():
                ctx_tok = vq.encode_indices(ctx).view(B, T, -1)  # (B,T,S)
                tgt_tok = vq.encode_indices(tgt).view(B, K, -1)  # (B,K,S)

            S = ctx_tok.shape[-1]
            all_tok = torch.cat([ctx_tok, tgt_tok], dim=1)       # (B,T+K,S)
            seq = all_tok.reshape(B, (T+K)*S).to(device)         # (B,L)

            x = seq[:, :-1]
            y = seq[:, 1:]

            logits = gpt(x)  # (B,L-1,V)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), cfg["train"]["grad_clip"])
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

        torch.save({"model": gpt.state_dict(), "cfg": cfg},
                   abs_path("checkpoints", f"gpt_epoch_{epoch}.pt"))

    torch.save({"model": gpt.state_dict(), "cfg": cfg}, abs_path("checkpoints", "gpt_final.pt"))
    print("Saved:", abs_path("checkpoints", "gpt_final.pt"))

if __name__ == "__main__":
    main()
