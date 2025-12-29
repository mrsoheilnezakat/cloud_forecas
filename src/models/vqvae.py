import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Downsample x8:
      256 -> 128 -> 64 -> 32
    Output z_e: (B, D, H/8, W/8)
    """
    def __init__(self, in_channels=1, hidden=128, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 4, 4, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 4, hidden // 2, 4, stride=2, padding=1),  # /4
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, hidden, 4, stride=2, padding=1),       # /8
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, embedding_dim, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """
    Upsample x8 to match encoder:
      32 -> 64 -> 128 -> 256
    """
    def __init__(self, out_channels=1, hidden=128, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden, hidden // 2, 4, stride=2, padding=1),  # x2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden // 2, hidden // 4, 4, stride=2, padding=1),  # x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden // 4, out_channels, 4, stride=2, padding=1),  # x8
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class VectorQuantizerEMA(nn.Module):
    """
    EMA VQ:
      - no one_hot
      - chunked argmin in fp32
    """
    def __init__(
        self,
        num_embeddings=1024,
        embedding_dim=64,
        commitment_cost=0.25,
        decay=0.99,
        eps=1e-5,
        dist_chunk=8192,
        dist_clamp=1.0e6,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.decay = float(decay)
        self.eps = float(eps)
        self.dist_chunk = int(dist_chunk)
        self.dist_clamp = float(dist_clamp)

        self.emb = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.emb.weight.data.normal_()

        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_w", self.emb.weight.data.clone())

    @torch.no_grad()
    def _nearest_indices_chunked(self, flat: torch.Tensor) -> torch.Tensor:
        # flat is fp32 (N,D)
        emb_w = self.emb.weight.float()  # (K,D)
        e2 = emb_w.pow(2).sum(dim=1).unsqueeze(0)  # (1,K)

        N, D = flat.shape
        out = torch.empty(N, device=flat.device, dtype=torch.long)

        for start in range(0, N, self.dist_chunk):
            end = min(start + self.dist_chunk, N)
            x = flat[start:end]  # (M,D)
            x2 = x.pow(2).sum(dim=1, keepdim=True)
            xe = x @ emb_w.t()
            dist = x2 - 2.0 * xe + e2
            dist = torch.clamp(dist, -self.dist_clamp, self.dist_clamp)
            out[start:end] = torch.argmin(dist, dim=1)
            del x, x2, xe, dist

        del emb_w, e2
        return out

    def forward(self, z_e):
        B, D, H, W = z_e.shape
        z = z_e.permute(0, 2, 3, 1).contiguous()      # (B,H,W,D)
        flat = z.view(-1, D).float()                   # (N,D)

        indices = self._nearest_indices_chunked(flat)  # (N,)
        z_q = self.emb(indices).view(B, H, W, D).float()

        if self.training:
            counts = torch.bincount(indices, minlength=self.num_embeddings).to(self.ema_cluster_size.dtype)
            self.ema_cluster_size.mul_(self.decay).add_(counts, alpha=(1.0 - self.decay))

            dw = torch.zeros(self.num_embeddings, D, device=flat.device, dtype=torch.float32)
            dw.index_add_(0, indices, flat)
            self.ema_w.mul_(self.decay).add_(dw, alpha=(1.0 - self.decay))

            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
            new_w = self.ema_w / cluster_size.unsqueeze(1)
            self.emb.weight.data.copy_(new_w.to(self.emb.weight.data.dtype))

        commit_loss = self.commitment_cost * F.mse_loss(z.float(), z_q.detach())

        z_q_st = z.float() + (z_q - z.float()).detach()
        z_q_st = z_q_st.permute(0, 3, 1, 2).contiguous()
        indices = indices.view(B, H, W)
        return z_q_st.to(z_e.dtype), commit_loss, indices


class VQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden=128, embedding_dim=64, num_embeddings=1024, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, hidden=hidden, embedding_dim=embedding_dim)
        self.vq = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=0.99,
            dist_chunk=8192,
        )
        self.decoder = Decoder(out_channels=in_channels, hidden=hidden, embedding_dim=embedding_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss, indices