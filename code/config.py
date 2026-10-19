#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全局参数配置加载器 —— 所有流水线脚本 (01~06) 统一从这里取参数。

用法:
    import config as C
    cfg = C.load()                      # 单例, 读 code/config.yaml
    base = cfg.base_dir                 # 项目根目录 deeplearning_model/
    lr   = cfg.training.lr              # 点式访问嵌套键
    p    = cfg.get_path("paths.uniprot_fasta")      # 相对路径自动拼 base_dir
    keep = cfg.get("gold_dataset.keep_columns")     # 任意深度取原始值

约定:
  * config.yaml 中【绝对路径】原样使用;【相对路径】相对项目根目录 (deeplearning_model/)。
  * 脚本内禁止再硬编码参数/路径, 一律走 cfg。
"""
import os
from typing import Any

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("config.py 需要 PyYAML，请先安装: pip install pyyaml") from e

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(CODE_DIR, "config.yaml")


class Config:
    """嵌套 dict 的点式访问包装 (cfg.training.lr / cfg.get_path(...))。"""

    def __init__(self, data: dict, base_dir: str):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "base_dir", base_dir)

    # ---- 点式访问 ----------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(f"config.yaml 中不存在键: {name!r}") from None
        return Config(value, self.base_dir) if isinstance(value, dict) else value

    def __getitem__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError:
            raise KeyError(f"config.yaml 中不存在键: {name!r}") from None
        return Config(value, self.base_dir) if isinstance(value, dict) else value

    def __contains__(self, name: str) -> bool:
        return name in self._data

    # ---- 通用取数 ----------------------------------------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """按点路径取原始值，如 cfg.get("training.lr")；缺失返回 default。"""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def get_path(self, dotted_key: str, must_exist: bool = False) -> str:
        """取路径键并解析为绝对路径 (相对路径拼 base_dir)。"""
        value = self.get(dotted_key)
        if value is None:
            raise KeyError(f"config.yaml 中不存在路径键: {dotted_key!r}")
        path = value if os.path.isabs(str(value)) else os.path.abspath(
            os.path.join(self.base_dir, str(value)))
        if must_exist and not os.path.exists(path):
            raise FileNotFoundError(f"配置路径不存在: {path} (键: {dotted_key})")
        return path

    def sidecar_npz(self, kind: str, feat_name: str) -> str:
        """返回某特征文件【配套】的 labels/split 路径。

        缺陷#4 引入「剔除缺失模态行」后，不同编码器的可特征化集合行数不再相同
        （fp/hybrid 399,528 vs unimol2 399,020），而 labels.npz/split.npz 只写一份，
        用后者索引前者会越界（实测 05_unimol_dde_mp512 触发 CUDA device-side assert）。
        故 04 为每个特征文件另存 labels_<stem>.npz / split_<stem>.npz；
        缺失时回落到全局路径以兼容旧产物。

        kind ∈ {"labels", "split"}
        """
        stem = os.path.splitext(feat_name)[0]          # features_esm.npz -> features_esm
        p = os.path.join(self.get_path("paths.features_dir"), f"{kind}_{stem}.npz")
        if os.path.exists(p):
            return p
        return self.get_path(f"paths.{kind}_npz")

    def as_dict(self) -> dict:
        return self._data

    # ---- 方向标签映射（唯一真相，缺陷#1）------------------------------------
    def direction_label_map(self) -> dict:
        """方向字符串 -> 训练用整数索引。

        全流程只从 config.yaml `training.direction_label_map` 读这一处定义。
        历史上外部打分脚本各自手写 {"activation":0,...}，与训练的
        {neutral:0, inhibition:1, activation:2} 相反，导致三个外部方向集结果被低估。
        """
        m = self.get("training.direction_label_map")
        if not isinstance(m, dict) or not m:
            raise KeyError("config.yaml 缺少 training.direction_label_map（方向标签唯一真相）")
        return {str(k): int(v) for k, v in m.items()}

    def direction_label_names(self) -> list:
        """按索引升序返回方向名称列表，供混淆矩阵刻度使用。

        以 direction_label_map 为准推导；若 config 同时给了 training.label_names，
        则校验二者一致（不一致直接报错，防止再次漂移）。
        """
        m = self.direction_label_map()
        idx_to_name = {v: k for k, v in m.items()}
        if len(idx_to_name) != len(m):
            raise ValueError(f"direction_label_map 的索引有重复: {m}")
        names = [idx_to_name.get(i, str(i)) for i in range(len(m))]
        declared = self.get("training.label_names")
        if declared:
            declared_list = [declared.get(str(i), names[i]) for i in range(len(m))]
            if declared_list != names:
                raise ValueError(
                    f"config.yaml 的 training.label_names={declared_list} 与 "
                    f"training.direction_label_map={m} 不一致，请统一（后者为准）")
        return names


_cache = None


def load(config_path: str = None, reload: bool = False) -> Config:
    """加载 config.yaml (默认单例缓存)。base_dir = 项目根目录 (code/ 的上一级)。"""
    global _cache
    if _cache is not None and not reload and config_path is None:
        return _cache
    path = os.path.abspath(config_path or DEFAULT_CONFIG_PATH)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config.yaml 顶层必须是映射 (mapping): {path}")
    base_dir = os.path.dirname(os.path.dirname(path))  # code/ -> 项目根
    cfg = Config(data, base_dir)
    if config_path is None:
        _cache = cfg
    return cfg


if __name__ == "__main__":
    c = load()
    print(f"base_dir      : {c.base_dir}")
    print(f"source_parquet: {c.get_path('paths.source_parquet')}")
    print(f"uniprot_fasta : {c.get_path('paths.uniprot_fasta')}")
    print(f"training.lr   : {c.training.lr}  epochs={c.training.epochs} bs={c.training.batch_size}")
    print(f"fp_bits       : {c.features.fp_bits}  radius={c.features.fp_radius}")
    print("config.yaml 加载正常 ✓")
