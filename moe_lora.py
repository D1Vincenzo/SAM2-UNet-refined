import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora
import math

# ====== 替代 loralib.Linear 的 SAMed 风格 LoRA 实现 ======
class _LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))  # Frozen during training
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.lora_a = nn.Linear(in_features, r, bias=False)
        self.lora_b = nn.Linear(r, out_features, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        delta = self.lora_b(self.lora_a(x))
        return base + delta


# ====== Dense MoE-LoRA Module ======
class MoELinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, n_experts=4, lambda_=0.01):
        super().__init__()
        self.lambda_ = lambda_
        self.n_experts = n_experts

        self.gate = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, n_experts)
        )

        self.experts = nn.ModuleList([
            _LoRALinear(in_features, out_features, r=r)
            for _ in range(n_experts)
        ])

    def forward(self, x):
        original_shape = x.shape
        reshaped = False

        if x.dim() == 4:
            B, H, W, D = x.shape
            x = x.view(B, H * W, D)
            reshaped = True
        elif x.dim() != 3:
            raise ValueError(f"[MoELinear] Unsupported input shape: {x.shape}")
        
        B, L, D = x.shape

        pooled = x.mean(dim=1)
        gate_logits = self.gate(pooled)
        weights = F.softmax(gate_logits, dim=-1)  # [B, n_experts]

        expert_outputs = [expert(x) for expert in self.experts]  # 每个 [B, L, out_features]
        stacked = torch.stack(expert_outputs, dim=1)  # [B, n_experts, L, out_features]
        weights = weights.view(B, self.n_experts, 1, 1)
        output = (stacked * weights).sum(dim=1)  # [B, L, out_features]

        if reshaped:
            output = output.view(B, H, W, -1)

        if self.training:
            prob_mean = weights.mean(dim=0).squeeze()
            entropy = - (prob_mean * torch.log(prob_mean + 1e-8)).sum()
            self.aux_loss = self.lambda_ * entropy

        return output


# ====== QKV 专用封装模块：输出 [B, L, 3*D] 或 [B, H, W, 3*D] ======
class MoEQKVLinear(nn.Module):
    def __init__(self, in_dim, out_features, r=4, n_experts=4, lambda_=0.01):
        super().__init__()
        self.moe = MoELinear(
            in_features=in_dim,
            out_features=out_features,
            r=r,
            n_experts=n_experts,
            lambda_=lambda_
        )

    def forward(self, x):
        # print("MoEQKVLinear got input shape:", x.shape)
        return self.moe(x)



# ====== 注入函数：替换 ViT 中的 qkv 层 ======
def replace_qkv_with_moe_qkv(model, r=4, n_experts=4):
    for name, module in model.named_modules():
        if name.endswith("attn.qkv") and isinstance(module, nn.Linear):
            parent = get_parent(model, name)
            attr = name.split('.')[-1]

            in_dim = module.in_features
            out_dim = module.out_features

            assert out_dim % 3 == 0, f"{name}: out_dim={out_dim} is not divisible by 3"
            true_dim = out_dim // 3

            print(f"[CHECK] {name}  in={in_dim}, out={out_dim} => q/k/v dim={true_dim}")

            # 替换为 MoEQKVLinear(in_dim, out_dim)
            setattr(parent, attr, MoEQKVLinear(in_dim, out_features=out_dim, r=r, n_experts=n_experts))
            print(f"[MoE-LoRA] Replaced {name} (Linear → MoEQKVLinear)")




def get_parent(model, name):
    parts = name.split('.')
    for p in parts[:-1]:
        model = getattr(model, p)
    return model
