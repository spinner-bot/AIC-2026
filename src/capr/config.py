"""配置加载。

职责：
    - 读取 YAML 配置
    - 支持点号路径访问（cfg.data.train_dir 等价于 cfg["data"]["train_dir"]）
    - 解析相对路径：以项目根（仓库根）为基准
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# 项目根：本文件位于 src/capr/config.py，向上三级即仓库根
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config:
    """支持 attribute 与 item 双访问的只读配置包装。"""

    def __init__(self, data: dict[str, Any], root: Path = _PROJECT_ROOT):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_root", root)

    def __getattr__(self, name: str) -> Any:
        try:
            val = self._data[name]
        except KeyError:
            raise AttributeError(f"配置项不存在: {name}") from None
        if isinstance(val, dict):
            return Config(val, self._root)
        return val

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def path(self, value: str | None) -> Path | None:
        """把配置中的路径字段解析为绝对路径（相对路径基于项目根）。"""
        if value is None:
            return None
        p = Path(value)
        return p if p.is_absolute() else self._root / p


def load_config(path: str | Path) -> Config:
    """加载 YAML 配置文件并返回 Config。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误: {path}")
    return Config(data)


def project_root() -> Path:
    return _PROJECT_ROOT


def set_seed(seed: int) -> None:
    """固定随机种子（python / numpy / torch / cuda），保证训练可复现。

    惰性 import 避免给 config 模块引入 torch 顶层依赖。若要 bit-exact 复现，
    还需额外设置 torch.backends.cudnn.deterministic=True（代价是训练变慢），
    竞赛的可复现复核只需结果级一致，故此处不强制开启。
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
