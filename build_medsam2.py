# build_medsam2.py
import torch
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

def build_medsam2(
    config_file: str,
    ckpt_path: str,
    device: str = "cuda",
    eval_mode: bool = True,
):
    # ✅ 初始化 Hydra 配置上下文
    with initialize(version_base=None, config_path="sam2_configs"):
        cfg = compose(config_name="sam2.1_hiera_t512.yaml")
        OmegaConf.resolve(cfg)

    # 实例化模型
    model = instantiate(cfg.model, _recursive_=True)

    # 加载 checkpoint
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt, strict=False)

    model = model.to(device)
    if eval_mode:
        model.eval()

    return model
