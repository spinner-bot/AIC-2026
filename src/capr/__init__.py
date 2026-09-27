"""CAPR-CLIP：面向含噪标签、细粒度、长尾图像分类的鲁棒微调方案。

V3 三层解耦架构：
    Stage 0  数据审计 + 防泄漏 group split
    Stage 1  冻结 CLIP 特征缓存
    Stage 2  q0 可靠度诊断（OOF / 鲁棒原型 / mutual-kNN / 条件 GMM）
    Stage 3  MVP 微调（LoRA + 可靠度加权 + 自适应锚定）
    Stage 5  延迟长尾校准（cRT / logit adjustment / BS-cRT）
"""

__version__ = "0.1.0"
