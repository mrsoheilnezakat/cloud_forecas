import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model=256, n_layers=6, n_heads=8, dropout=0.1, max_len=4096):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(int(max_len), self.d_model)
        self.drop = nn.Dropout(float(dropout))

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=int(n_heads),
            dim_feedforward=4*self.d_model,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.ln = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, idx):
        """
        idx: (B, L)
        logits: (B, L, V)
        """
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        # causal mask: True means "blocked"
        causal = torch.triu(torch.ones(L, L, device=idx.device, dtype=torch.bool), diagonal=1)
        x = self.tr(x, mask=causal)

        x = self.ln(x)
        return self.head(x)