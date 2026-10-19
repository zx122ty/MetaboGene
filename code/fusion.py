#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FusionModule — DualTaskMLP 的模态融合层（concat / bilinear / gated / xattn）。

背景（handover: prompt/26-10-13-staging/01-改为attention融合-handover.md）：
原 DualTaskMLP 把 [mol | prot]（可选各自线性投影后）concat 喂共享 trunk。
本模块把该融合点参数化，供受控消融：数据/特征/split/seed/超参/双任务头/训练协议全不变。

公式（最终实现，报告 prompt/26-10-13/02-fusion-结果.md 写清）：
  统一两段式：先各自投影到工作维 u_m = Linear_m(mol), u_p = Linear_p(prot) ∈ R^work_dim，
  再按 kind 融合到 out_dim（=trunk 输入维）：

  concat   : 直接 torch.cat([mol, prot], dim=1)（无参数）。主脚本中 concat 走原路径、
             不经本模块（回归安全）；此实现仅作单元测试参照。
  bilinear : h_m = Linear(u_m→out_dim), h_p = Linear(u_p→out_dim)
             z   = Linear(3·out_dim → out_dim)( concat(h_m, h_p, h_m ⊙ h_p) )
             （Hadamard 积显式建模两模态逐维交互，比纯 concat 多一个交互通道。）
  gated    : h_m, h_p 同上
             g   = sigmoid( Linear(2·out_dim → out_dim)( concat(h_m, h_p) ) )
             z   = g ⊙ h_m + (1−g) ⊙ h_p
             （逐维软选择"该信分子还是蛋白"。）
  xattn    : pseudo-token 双向 cross-attention（单向量模态不 token 化会退化成标量门控）：
             u_m/u_p ∈ R^work_dim reshape 成 T=xattn_tokens 个 token（token_dim = work_dim // T）
             方向1: Q=mol-tokens, K/V=prot-tokens；方向2: Q=prot-tokens, K/V=mol-tokens
             nn.MultiheadAttention(embed=token_dim, heads=xattn_heads, batch_first=True)
             各方向 residual + LayerNorm 后时间维 mean-pool → concat(两方向) → Linear(→out_dim)
  ---- v2 raw-dims 版（26-10-13 追加，*消除 work_dim 瓶颈混淆项*）----
  输入与基线完全一致：m = mol raw（不压缩），p = _split 已完成 proj_esm 的蛋白块
  （与基线同为 1024）。基线 concat 通道 [m; p] 原样保留，融合仅"追加"交互机制：

  bilinear_raw : z = concat(m, p, m ⊙ W(p))，W: dp→dm（交互在 mol 空间）→ out = 2·dm + dp
  gated_raw    : g_m = sigmoid(W_m[m;p])→dm，g_p = sigmoid(W_p[m;p])→dp
                 z = concat(g_m⊙m, g_p⊙p) → out = dm + dp（trunk 输入与基线同宽）
  xattn_raw    : m→T token×(dm/T)，p→T token×(dp/T)，各自线性投影到 embed=work_dim 空间，
                 双向 MHA（residual 在 Q 空间 + LN + mean-pool），
                 z = concat(m, p, pool(m←p), pool(p←m)) → out = dm + dp + 2·embed

  退化     : dp==0（"仅分子特征"消融路径）时，全部非 concat kind 统一退化为
             Linear(dm→out_dim)(mol)，不创建交互参数。

融合内部不加 Dropout/激活（共享 trunk 已有 ReLU+Dropout×2），保持最小改动、利于归因。

自检（python code/fusion.py）：
  * 四种 kind 输出形状 [B, out_dim]（concat 为 [B, dm+dp]）/ 无 NaN / 反传冒烟；
  * work_dim 不整除 xattn_tokens 时报错；
  * dp==0 退化路径；
  * concat 回归：内嵌旧版 DualTaskMLP（改动前逐字拷贝）与改后 05_train_evaluate.DualTaskMLP
    (fusion="concat") 同 seed 构建，state_dict 结构与 forward 输出逐位一致。
"""
import importlib.util
import os

import torch
import torch.nn as nn

import config as C

cfg = C.load()

KINDS = ("concat", "bilinear", "gated", "xattn",
         "bilinear_raw", "gated_raw", "xattn_raw")
RAW_KINDS = ("bilinear_raw", "gated_raw", "xattn_raw")


class FusionModule(nn.Module):
    """双模态（单向量）融合：mol:[B,dm] + prot:[B,dp] → fused:[B,out_dim]。"""

    def __init__(self, kind: str, dm: int, dp: int, work_dim: int, out_dim: int,
                 xattn_tokens: int = 4, xattn_heads: int = 4):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"未知融合类型: {kind!r}（可选 {KINDS}）")
        self.kind = kind
        self.dm, self.dp, self.work_dim, self.out_dim = dm, dp, work_dim, out_dim
        self.degraded = (dp == 0)  # 仅分子消融：无蛋白模态
        if kind == "concat":       # 参照实现（无参数）；主脚本 concat 走原路径不经此
            return
        if self.degraded:
            self.mol_only = nn.Linear(dm, out_dim)
            return
        if kind in RAW_KINDS:
            # ---- v2 (26-10-13 追加) raw-dims 无瓶颈融合 ----
            # dm/dp = 与基线完全相同的输入维（mol raw 2048 不压缩；p 已在 _split 完成
            # proj_esm，与基线一致）；基线 concat 通道 [m; p] 原样保留，仅追加交互项。
            if kind == "bilinear_raw":
                self.w_prot2mol = nn.Linear(dp, dm)      # prot → mol 空间
                self.out_dim = 2 * dm + dp               # [m; p; m ⊙ W(p)]
            elif kind == "gated_raw":
                self.gate_m = nn.Linear(dm + dp, dm)     # mol 逐维门控
                self.gate_p = nn.Linear(dm + dp, dp)     # prot 逐维门控
                self.out_dim = dm + dp                   # 与基线 trunk 输入同宽
            else:  # xattn_raw
                if dm % xattn_tokens != 0 or dp % xattn_tokens != 0:
                    raise ValueError(
                        f"xattn_raw: dm={dm}, dp={dp} 必须都能被 xattn_tokens={xattn_tokens} 整除")
                if work_dim % xattn_heads != 0:
                    raise ValueError(
                        f"xattn_raw: work_dim={work_dim} (注意力 embed) 必须能被 "
                        f"xattn_heads={xattn_heads} 整除")
                self.tokens = xattn_tokens
                self.tok_m, self.tok_p = dm // xattn_tokens, dp // xattn_tokens
                self.embed = work_dim
                # 两模态 token 维不同 → 各自线性投影到同一 embed 空间再注意力
                self.q_m = nn.Linear(self.tok_m, self.embed)
                self.kv_p = nn.Linear(self.tok_p, 2 * self.embed)
                self.q_p = nn.Linear(self.tok_p, self.embed)
                self.kv_m = nn.Linear(self.tok_m, 2 * self.embed)
                self.attn_mp = nn.MultiheadAttention(self.embed, xattn_heads, batch_first=True)
                self.attn_pm = nn.MultiheadAttention(self.embed, xattn_heads, batch_first=True)
                self.ln_m = nn.LayerNorm(self.embed)
                self.ln_p = nn.LayerNorm(self.embed)
                self.out_dim = dm + dp + 2 * self.embed  # [m; p; pool(m←p); pool(p←m)]
            return
        self.u_m = nn.Linear(dm, work_dim)
        self.u_p = nn.Linear(dp, work_dim)
        if kind == "bilinear":
            self.h_m = nn.Linear(work_dim, out_dim)
            self.h_p = nn.Linear(work_dim, out_dim)
            self.fuse = nn.Linear(3 * out_dim, out_dim)
        elif kind == "gated":
            self.h_m = nn.Linear(work_dim, out_dim)
            self.h_p = nn.Linear(work_dim, out_dim)
            self.gate = nn.Linear(2 * out_dim, out_dim)
        else:  # xattn
            if work_dim % xattn_tokens != 0:
                raise ValueError(
                    f"xattn: work_dim={work_dim} 必须能被 xattn_tokens={xattn_tokens} 整除")
            self.tokens = xattn_tokens
            self.token_dim = work_dim // xattn_tokens
            if self.token_dim % xattn_heads != 0:
                raise ValueError(
                    f"xattn: token_dim={self.token_dim} (work_dim/xattn_tokens) "
                    f"必须能被 xattn_heads={xattn_heads} 整除")
            self.attn_mp = nn.MultiheadAttention(self.token_dim, xattn_heads, batch_first=True)
            self.attn_pm = nn.MultiheadAttention(self.token_dim, xattn_heads, batch_first=True)
            self.ln_m = nn.LayerNorm(self.token_dim)
            self.ln_p = nn.LayerNorm(self.token_dim)
            self.fuse = nn.Linear(2 * self.token_dim, out_dim)

    def forward(self, mol: torch.Tensor, prot: torch.Tensor) -> torch.Tensor:
        if self.kind == "concat":
            return torch.cat([mol, prot], dim=1)
        if self.degraded:
            return self.mol_only(mol)
        if self.kind == "bilinear_raw":
            inter = mol * self.w_prot2mol(prot)          # 交互项（mol 空间，2048 维）
            return torch.cat([mol, prot, inter], dim=1)  # 基线通道 [m; p] 原样保留
        if self.kind == "gated_raw":
            mp = torch.cat([mol, prot], dim=1)
            g_m = torch.sigmoid(self.gate_m(mp))
            g_p = torch.sigmoid(self.gate_p(mp))
            return torch.cat([g_m * mol, g_p * prot], dim=1)  # 宽度与基线一致
        if self.kind == "xattn_raw":
            b, t = mol.shape[0], self.tokens
            m_tok = mol.view(b, t, self.tok_m)
            p_tok = prot.view(b, t, self.tok_p)
            q_m, q_p = self.q_m(m_tok), self.q_p(p_tok)          # [B,T,embed]
            k_p, v_p = self.kv_p(p_tok).chunk(2, dim=-1)
            k_m, v_m = self.kv_m(m_tok).chunk(2, dim=-1)
            am, _ = self.attn_mp(q_m, k_p, v_p)                  # mol tokens ← prot tokens
            ap, _ = self.attn_pm(q_p, k_m, v_m)                  # prot tokens ← mol tokens
            vm = self.ln_m(am + q_m).mean(dim=1)                 # residual 在 Q 空间 + LN + pool
            vp = self.ln_p(ap + q_p).mean(dim=1)
            return torch.cat([mol, prot, vm, vp], dim=1)         # 基线通道原样保留
        u_m, u_p = self.u_m(mol), self.u_p(prot)
        if self.kind == "bilinear":
            h_m, h_p = self.h_m(u_m), self.h_p(u_p)
            return self.fuse(torch.cat([h_m, h_p, h_m * h_p], dim=1))
        if self.kind == "gated":
            h_m, h_p = self.h_m(u_m), self.h_p(u_p)
            g = torch.sigmoid(self.gate(torch.cat([h_m, h_p], dim=1)))
            return g * h_m + (1.0 - g) * h_p
        # xattn: pseudo-token 双向 cross-attention
        b, t, d = u_m.shape[0], self.tokens, self.token_dim
        tm, tp = u_m.view(b, t, d), u_p.view(b, t, d)
        am, _ = self.attn_mp(tm, tp, tp)    # mol tokens ← prot tokens
        ap, _ = self.attn_pm(tp, tm, tm)    # prot tokens ← mol tokens
        vm = self.ln_m(am + tm).mean(dim=1)  # residual + LN + mean-pool → [B, token_dim]
        vp = self.ln_p(ap + tp).mean(dim=1)
        return self.fuse(torch.cat([vm, vp], dim=1))


# ============================ 自检 ============================

class _DualTaskMLP_Legacy(nn.Module):
    """改版前的 DualTaskMLP（回归参照，逐字拷贝自 05_train_evaluate.py 改动前版本）。"""

    def __init__(self, in_dim, hidden=None, proj_esm=0, protein_dim=None, fp_dim=None,
                 mol_proj=0):
        super().__init__()
        if hidden is None:
            hidden = cfg.get("training.hidden_dim")
        dropout = cfg.get("training.dropout")
        if fp_dim is None:
            fp_dim = cfg.get("features.fp_bits")
        self.fp_dim = fp_dim
        self.protein_dim = in_dim - fp_dim if protein_dim is None else protein_dim
        self.proj_esm = proj_esm
        self.mol_proj = mol_proj
        if mol_proj > 0:
            self.mol_proj_layer = nn.Linear(fp_dim, mol_proj)
            eff_mol = mol_proj
        else:
            eff_mol = fp_dim
        if proj_esm > 0:
            self.esm_proj = nn.Linear(self.protein_dim, proj_esm)
            eff_prot = proj_esm
        else:
            eff_prot = self.protein_dim
        eff_in = eff_mol + eff_prot
        self.fp = nn.Sequential(
            nn.Linear(eff_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(hidden, 1)
        self.dir_head = nn.Linear(hidden, 3)

    def _split(self, x):
        mol = x[:, :self.fp_dim]
        prot = x[:, self.fp_dim:self.fp_dim + self.protein_dim]
        if self.mol_proj > 0:
            mol = self.mol_proj_layer(mol)
        if self.proj_esm > 0:
            prot = self.esm_proj(prot)
        return mol, prot

    def forward(self, x):
        if self.proj_esm > 0 or self.mol_proj > 0:
            mol, prot = self._split(x)
            x = torch.cat([mol, prot], dim=1)
        h = self.fp(x)
        return self.binary_head(h).squeeze(-1), self.dir_head(h)


def _load_train_module():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_train_evaluate.py")
    spec = importlib.util.spec_from_file_location("train_eval_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    dm, dp, wd, od, T = 2048, 2560, 256, 512, 4
    torch.manual_seed(0)
    mol = torch.randn(32, dm)
    prot = torch.randn(32, dp)
    print("== FusionModule 单元自检 ==")
    print("  -- v1（work_dim 瓶颈版, dp=2560）--")
    for kind in ("concat", "bilinear", "gated", "xattn"):
        torch.manual_seed(0)
        f = FusionModule(kind, dm, dp, wd, od, xattn_tokens=T)
        expected = dm + dp if kind == "concat" else od
        out = f(mol, prot)
        assert out.shape == (32, expected), (kind, tuple(out.shape))
        assert torch.isfinite(out).all(), kind
        n_par = sum(p.numel() for p in f.parameters())
        if n_par > 0:
            out.sum().backward()  # 反传冒烟（concat 无参数、无 grad_fn，跳过）
        print(f"  {kind:9s} out={tuple(out.shape)}  params={n_par:,}  ✓")
    print("  -- v2 raw（无瓶颈, dp=1024 与基线 proj_esm 1024 一致）--")
    dp_raw = 1024
    raw_expected = {"bilinear_raw": 2 * dm + dp_raw,      # [m; p; m⊙W(p)]
                    "gated_raw": dm + dp_raw,             # [g_m⊙m; g_p⊙p] 与基线同宽
                    "xattn_raw": dm + dp_raw + 2 * wd}    # [m; p; pool_m; pool_p]
    prot_raw = torch.randn(32, dp_raw)
    for kind in RAW_KINDS:
        torch.manual_seed(0)
        f = FusionModule(kind, dm, dp_raw, wd, od, xattn_tokens=T)
        out = f(mol, prot_raw)
        assert out.shape == (32, raw_expected[kind]), (kind, tuple(out.shape))
        assert torch.isfinite(out).all(), kind
        n_par = sum(p.numel() for p in f.parameters())
        out.sum().backward()
        print(f"  {kind:9s} out={tuple(out.shape)}  params={n_par:,}  ✓")
    try:
        FusionModule("xattn", dm, dp, 100, od, xattn_tokens=4)
        raise SystemExit("xattn 整除校验未生效 ✗")
    except ValueError:
        print("  xattn work_dim 整除校验 ✓")
    for kind in ("bilinear", "gated", "xattn", "bilinear_raw", "gated_raw", "xattn_raw"):
        torch.manual_seed(0)
        f = FusionModule(kind, dm, 0, wd, od, xattn_tokens=T)
        out = f(mol, torch.zeros(32, 0))
        assert out.shape == (32, od) and torch.isfinite(out).all(), kind
        print(f"  退化(dp=0) {kind:9s} out={tuple(out.shape)}  ✓")

    # concat 回归：新旧 DualTaskMLP 同 seed 构建 → state_dict 结构与 forward 逐位一致
    mod = _load_train_module()
    x = torch.randn(64, dm + dp)
    torch.manual_seed(42)
    old = _DualTaskMLP_Legacy(dm + dp, proj_esm=1024, protein_dim=dp, fp_dim=dm)
    torch.manual_seed(42)
    new = mod.DualTaskMLP(dm + dp, proj_esm=1024, protein_dim=dp, fp_dim=dm, fusion="concat")
    assert list(old.state_dict().keys()) == list(new.state_dict().keys()), \
        (list(old.state_dict().keys()), list(new.state_dict().keys()))
    old.eval(), new.eval()
    with torch.no_grad():
        ob, odir = old(x)
        nb, ndir = new(x)
    assert torch.equal(ob, nb) and torch.equal(odir, ndir), "concat forward 与旧版不一致 ✗"
    print("  concat 回归：state_dict 结构 + forward 输出与旧版逐位一致 ✓")
    print("全部自检通过 ✓")


if __name__ == "__main__":
    main()
