"""环境验证：CUDA 可用性 + 关键依赖 + CLIP 权重加载。

用法：
    D:\\anaconda\\envs\\aic\\python.exe scripts/verify_env.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    import torch

    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "CUDA 不可用！"
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {torch.cuda.get_device_name(0)}  显存 {props.total_memory / 1e9:.1f} GB")

    # 实际跑一个 GPU 张量，确认 CUDA 运行时正常
    x = torch.randn(64, 64, device="cuda")
    y = (x @ x).sum()
    torch.cuda.synchronize()
    print(f"  CUDA 前向测试: {float(y):.4f}")

    import faiss
    import imagehash
    import transformers

    print(f"faiss {faiss.__version__}  imagehash {imagehash.__version__}  transformers {transformers.__version__}")

    from transformers import CLIPModel, CLIPProcessor

    from src.capr.config import project_root

    model_dir = project_root() / "models" / "clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_dir)
    cfg = model.config
    print(
        f"CLIP 加载成功  projection_dim={cfg.projection_dim}  "
        f"vision_layers={cfg.vision_config.num_hidden_layers}  patch_size={cfg.vision_config.patch_size}"
    )
    print("ENV OK")


if __name__ == "__main__":
    main()
