"""模型构建。

职责:
    - 加载 CLIP ViT-B/32（HF transformers 的 OpenAI 官方权重）
    - PEFT 模块：CoOp（prompt tuning）/ LoRA / Adapter / Linear Probe
    - 前向 + 分类头
"""

# TODO: 实现 build_model(backbone, peft, num_classes) -> model
