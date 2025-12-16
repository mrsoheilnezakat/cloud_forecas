import os, random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def project_root() -> str:
    # works when running scripts from project root
    return os.getcwd()

def abs_path(*parts) -> str:
    return os.path.abspath(os.path.join(project_root(), *parts))

def ensure_dir(path: str):
    """
    Create directory if missing.
    If a FILE exists with same name, raise a clear error.
    """
    path = os.path.abspath(path)
    if os.path.exists(path) and not os.path.isdir(path):
        raise RuntimeError(f"Cannot create directory '{path}': a FILE with the same name already exists.")
    os.makedirs(path, exist_ok=True)

@torch.no_grad()
def top_k_sample(logits: torch.Tensor, top_k: int = 0, temperature: float = 1.0):
    """
    logits: (B, V)
    returns: (B,) sampled token indices
    """
    logits = logits / max(float(temperature), 1e-8)

    if top_k and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        v, _ = torch.topk(logits, k, dim=-1)
        cutoff = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < cutoff, torch.full_like(logits, -1e10), logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)
