import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml, glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from src.models.vqvae import VQVAE
from src.utils import ensure_dir, abs_path

@torch.no_grad()
def main():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    vq = VQVAE(**cfg["vqvae"]).to(device)
    vq.load_state_dict(torch.load(abs_path("checkpoints","vqvae_final.pt"), map_location=device)["model"])
    vq.eval()

    image_size = cfg["data"]["image_size"]
    tfm = T.Compose([T.Resize((image_size,image_size)), T.ToTensor()])

    p = sorted(glob.glob(r"data\val\*\*.png"))[0]
    x = tfm(Image.open(p).convert("L")).unsqueeze(0).to(device)

    x_hat, _, _ = vq(x)
    x_hat = x_hat[0,0].cpu().numpy()
    x_in  = x[0,0].cpu().numpy()

    ensure_dir(abs_path("outputs"))
    Image.fromarray((x_in*255).astype(np.uint8)).save(abs_path("outputs","recon_input.png"))
    Image.fromarray((np.clip(x_hat,0,1)*255).astype(np.uint8)).save(abs_path("outputs","recon_output.png"))
    print("Saved outputs\\recon_input.png and outputs\\recon_output.png")

if __name__ == "__main__":
    main()
