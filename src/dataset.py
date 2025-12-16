import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class SequenceFolderDataset(Dataset):
    """
    Expects:
      root_dir/seq_xxxx/000.png,001.png,...

    Returns:
      context: (T,1,H,W)
      target : (K,1,H,W)
    """
    def __init__(self, root_dir: str, image_size: int, context_frames: int, pred_frames: int, stride: int = 1):
        self.root_dir = root_dir
        self.context_frames = int(context_frames)
        self.pred_frames = int(pred_frames)
        self.stride = int(stride)

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # grayscale => (1,H,W) in [0,1]
        ])

        self.seqs = sorted([p for p in glob(os.path.join(root_dir, "*")) if os.path.isdir(p)])
        self.samples = []

        needed = (self.context_frames + self.pred_frames) * self.stride

        for seq in self.seqs:
            frames = sorted(glob(os.path.join(seq, "*.png")))
            if len(frames) < needed:
                continue

            max_start = len(frames) - needed
            for s in range(0, max_start + 1):
                idxs = [s + i * self.stride for i in range(self.context_frames + self.pred_frames)]
                self.samples.append((frames, idxs))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No training samples found in '{root_dir}'. "
                f"Check folder structure and that each sequence has enough frames."
            )

    def __len__(self):
        return len(self.samples)

    def _load(self, path: str):
        img = Image.open(path).convert("L")
        return self.transform(img)

    def __getitem__(self, idx: int):
        frames, idxs = self.samples[idx]
        paths = [frames[i] for i in idxs]
        xs = [self._load(p) for p in paths]

        context = torch.stack(xs[:self.context_frames], dim=0)     # (T,1,H,W)
        target  = torch.stack(xs[self.context_frames:], dim=0)     # (K,1,H,W)
        return context, target
