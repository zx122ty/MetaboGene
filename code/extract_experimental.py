#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提取 experimental 行 —— 参数统一来自 config.yaml (paths.source_parquet / paths.experimental_csv)"""
import pandas as pd

import config as C

cfg = C.load()


def main():
    src = cfg.get_path("paths.source_parquet")
    out = cfg.get_path("paths.experimental_csv")

    # 读取 parquet 文件
    df = pd.read_parquet(src)

    # 筛选 experimental_method_role == "experimental" 的行
    df_exp = df[df["experimental_method_role"] == "experimental"].copy()

    # 保存为 CSV，UTF-8 BOM 编码
    df_exp.to_csv(out, index=False, encoding="utf-8-sig")

    # 打印行数
    print(f"原数据行数: {len(df)}")
    print(f"筛选后行数: {len(df_exp)}")


if __name__ == "__main__":
    main()
