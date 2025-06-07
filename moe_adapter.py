import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.prompt = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        return x + self.prompt(x)


class MoEAdapterBlock(nn.Module):
    def __init__(self, shared_block: nn.Module, n_experts=4, lambda_=0.01):
        super().__init__()
        self.block = shared_block
        self.lambda_ = lambda_
        self.n_experts = n_experts
        dim = shared_block.attn.qkv.in_features

        self.gate = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_experts)
        )

        self.experts = nn.ModuleList([
            AdapterExpert(dim) for _ in range(n_experts)
        ])

    def forward(self, x):
        reshaped = False
        if x.dim() == 4:
            B, H, W, D = x.shape
            x = x.view(B, H * W, D)
            reshaped = True
        elif x.dim() != 3:
            raise ValueError(f"[MoEAdapterBlock] Unsupported input shape: {x.shape}")

        B, L, D = x.shape
        pooled = x.mean(dim=1)
        gate_logits = self.gate(pooled)  # [B, n_experts]
        weights = F.softmax(gate_logits, dim=-1)  # [B, n_experts]

        # expert outputs
        outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, n_experts, L, D]
        weights = weights.view(B, self.n_experts, 1, 1)
        x = (outputs * weights).sum(dim=1)  # [B, L, D]

        if reshaped:
            x = x.view(B, H, W, D)

        # 通入共享 block
        x = self.block(x)

        # aux loss: encourage diverse usage
        if self.training:
            prob_mean = weights.mean(dim=0).squeeze()
            entropy = - (prob_mean * torch.log(prob_mean + 1e-8)).sum()
            self.aux_loss = self.lambda_ * entropy

        return x
