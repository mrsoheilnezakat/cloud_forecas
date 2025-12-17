import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_ch=1, hidden=128, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden//2, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(hidden//2, hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hidden, emb_dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, hidden=128, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(emb_dim, hidden, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden//2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(hidden//2, hidden//4, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(hidden//4, out_ch, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)

        self.emb = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.emb.weight.data.uniform_(-1.0/self.num_embeddings, 1.0/self.num_embeddings)

    def forward(self, z_e):
        B, D, H, W = z_e.shape
        z = z_e.permute(0,2,3,1).contiguous()      # (B,H,W,D)
        flat = z.view(-1, D)                       # (BHW, D)

        emb_w = self.emb.weight                    # (K,D)
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ emb_w.t()
                + emb_w.pow(2).sum(1, keepdim=True).t())  # (BHW,K)

        indices = torch.argmin(dist, dim=1)        # (BHW,)
        z_q = self.emb(indices).view(B, H, W, D)

        # losses
        z_q_detached = z_q.detach()
        z_detached = z.detach()
        loss = F.mse_loss(z_q_detached, z_detached) + self.commitment_cost * F.mse_loss(z_q, z_detached)

        # straight-through estimator
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0,3,1,2).contiguous()    # (B,D,H,W)
        indices = indices.view(B, H, W)
        return z_q, loss, indices

class VQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden=128, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(in_ch=in_channels, hidden=hidden, emb_dim=embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(out_ch=in_channels, hidden=hidden, emb_dim=embedding_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_hat = self.decoder(z_q)
        # return x_hat, vq_loss, indices
        return x_hat, 0.001 * vq_loss, indices

    @torch.no_grad()
    def encode_indices(self, x):
        z_e = self.encoder(x)
        _, _, indices = self.vq(z_e)
        return indices  # (B,Htok,Wtok)

    @torch.no_grad()
    def decode_indices(self, indices):
        z_q = self.vq.emb(indices)                 # (B,H,W,D)
        z_q = z_q.permute(0,3,1,2).contiguous()    # (B,D,H,W)
        return self.decoder(z_q)
