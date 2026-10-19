#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8 — 外部数据准备：Reactome 调控方向事件（经 ContentService API）

背景：download.reactome.org（S3）对本机 IP 拒绝（403 AccessDenied），但
reactome.org/ContentService API 可达。因此不下载 BioPAX OWL，改为走 API
遍历人类 reaction，提取"小分子调控方向"事件：

  reaction.regulatedBy[] → Regulation 事件（schemaClass = Positive/NegativeRegulation
  = 方向）→ regulator 物理实体（SmallMolecule 的 ChEBI id / Complex 递归展开）；
  pair 的蛋白侧取该 reaction 的 catalystActivity 的物理实体（EWAS→UniProt，
  Complex/DefinedSet 递归展开）。

产物 (data/external/reactome/):
  events_hierarchy.json   人类事件树缓存
  reactions_compact.jsonl phase1: {reaction_stId: {reg:[...], cat:[pe-stid]}}（可断点续跑）
  regulations.jsonl       phase2: Regulation 事件对象
  entities.jsonl          phase3: 展开的物理实体（Complex/DefinedSet/EWAS/SmallMolecule）
  regulation_edges.tsv    最终边: chebi_id \t direction \t reaction \t uniprot_set

用法: python code/08_external_reactome_prep.py [--workers 8] [--max-phase 3]
"""
import argparse
import collections
import json
import os
import sys
import time

import requests

import config as C

cfg = C.load()
OUT = os.path.join(cfg.get_path("paths.external_dir"), "reactome")
os.makedirs(OUT, exist_ok=True)

BASE = "https://reactome.org/ContentService/data/query"
UA = {"User-Agent": "Mozilla/5.0 (metabo-gene-research)"}
SESSION = requests.Session()
SESSION.headers.update(UA)


def fetch(stid, retries=4):
    """GET 一个对象；失败返回 None（调用方记日志）。"""
    for a in range(retries):
        try:
            r = SESSION.get(f"{BASE}/{stid}", timeout=40)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            time.sleep(1.5 * (a + 1))
        except requests.RequestException:
            time.sleep(1.5 * (a + 1))
    return None


def load_jsonl(path):
    if not os.path.exists(path):
        return {}
    d = {}
    with open(path) as f:
        for line in f:
            try:
                o = json.loads(line)
                d[o["stId"]] = o
            except Exception:
                pass
    return d


def append_jsonl(path, objs):
    with open(path, "a") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- phase 0
def phase0_hierarchy():
    hp = os.path.join(OUT, "events_hierarchy.json")
    if os.path.exists(hp) and os.path.getsize(hp) > 1000:
        d = json.load(open(hp))
    else:
        r = SESSION.get("https://reactome.org/ContentService/data/eventsHierarchy/9606",
                        timeout=120)
        r.raise_for_status()
        d = r.json()
        json.dump(d, open(hp, "w"))
    rxns = []

    def walk(nodes):
        for n in nodes:
            if n.get("type") == "Reaction":
                rxns.append(n["stId"])
            if "children" in n:
                walk(n["children"])
    walk(d)
    print(f"[8-reactome] human reactions: {len(rxns):,}")
    return rxns


# ---------------------------------------------------------------- phase 1
def phase1_reactions(rxns, workers=8):
    """逐 reaction 取 regulatedBy / catalystActivity 紧凑视图。"""
    done = load_jsonl(os.path.join(OUT, "reactions_compact.jsonl"))
    todo = [r for r in rxns if r not in done]
    print(f"[8-reactome] phase1 reactions: {len(todo):,} todo / {len(done):,} cached")

    import concurrent.futures as cf
    buf, t0 = [], time.time()

    def one(stid):
        j = fetch(stid)
        if j is None:
            return None
        reg = [e.get("stId") for e in (j.get("regulatedBy") or []) if e.get("stId")]
        cat = [(ca.get("physicalEntity") or {}).get("stId")
               for ca in (j.get("catalystActivity") or [])]
        cat = [c for c in cat if c]
        return {"stId": stid, "reg": reg, "cat": cat}

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, o in enumerate(ex.map(one, todo)):
            if o is not None:
                buf.append(o)
            if len(buf) >= 500:
                append_jsonl(os.path.join(OUT, "reactions_compact.jsonl"), buf)
                buf = []
            if (i + 1) % 500 == 0:
                print(f"  phase1 {i+1:,}/{len(todo):,}  ({time.time()-t0:.0f}s)", flush=True)
    if buf:
        append_jsonl(os.path.join(OUT, "reactions_compact.jsonl"), buf)
    done = load_jsonl(os.path.join(OUT, "reactions_compact.jsonl"))
    reg_ids = sorted({s for o in done.values() for s in o["reg"]})
    cat_ids = sorted({s for o in done.values() for s in o["cat"]})
    print(f"[8-reactome] phase1 done: {len(done):,} reactions, "
          f"{len(reg_ids):,} regulation events, {len(cat_ids):,} catalyst PEs")
    return reg_ids, cat_ids


# ---------------------------------------------------------------- phase 2
def phase2b_catalysts(workers=8):
    """默认 query JSON 不含 catalystActivity → 走属性端点:
    /query/{rxn}/catalystActivity → CatalystActivity dbId → /query/{caId}/physicalEntity → PE stId。
    只对有调控记录的 reaction 做（催化剂仅在组边时需要）。产物 catalysts.jsonl。"""
    cat_path = os.path.join(OUT, "catalysts.jsonl")
    done = load_jsonl(cat_path)
    reactions = load_jsonl(os.path.join(OUT, "reactions_compact.jsonl"))
    hosts = sorted({rxn for rxn, o in reactions.items() if o["reg"]})
    todo = [r for r in hosts if r not in done]
    print(f"[8-reactome] phase2b catalysts: {len(todo):,} todo / {len(done):,} cached")

    import concurrent.futures as cf
    buf, t0 = [], time.time()

    def one(rxn):
        def txt(url):
            for a in range(4):
                try:
                    r = SESSION.get(url, timeout=40)
                    if r.status_code == 200:
                        return r.text
                    if r.status_code == 404:
                        return ""
                except requests.RequestException:
                    pass
                time.sleep(1.5 * (a + 1))
            return ""
        pes = []
        txt_ca = txt(f"{BASE}/{rxn}/catalystActivity")
        for line in txt_ca.splitlines():
            parts = line.split("\t")
            if not parts or not parts[0].strip():
                continue
            ca_id = parts[0].strip()
            txt_pe = txt(f"{BASE}/{ca_id}/physicalEntity")
            first = txt_pe.splitlines()[0].split("\t")[0].strip() if txt_pe.strip() else ""
            if first:
                pes.append(first)
        return {"stId": rxn, "cat": sorted(set(pes))}

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, o in enumerate(ex.map(one, todo)):
            buf.append(o)
            if len(buf) >= 300:
                append_jsonl(cat_path, buf)
                buf = []
            if (i + 1) % 300 == 0:
                print(f"  phase2b {i+1:,}/{len(todo):,}  ({time.time()-t0:.0f}s)", flush=True)
    if buf:
        append_jsonl(cat_path, buf)
    n = sum(len(load_jsonl(cat_path)[r]["cat"]) for r in load_jsonl(cat_path))
    print(f"[8-reactome] phase2b done: catalyst PEs collected over {len(load_jsonl(cat_path)):,} reactions")
    return None


def phase2_regulations(reg_ids, workers=8):
    done = load_jsonl(os.path.join(OUT, "regulations.jsonl"))
    todo = [r for r in reg_ids if r not in done]
    print(f"[8-reactome] phase2 regulations: {len(todo):,} todo / {len(done):,} cached")

    import concurrent.futures as cf
    buf, t0 = [], time.time()

    def one(stid):
        j = fetch(stid)
        if j is None:
            return None
        reg = j.get("regulator") or {}
        return {"stId": stid, "schemaClass": j.get("schemaClass"),
                "regulator": reg}  # regulator 原样保存（含 identifier / hasComponent 引用）

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, o in enumerate(ex.map(one, todo)):
            if o is not None:
                buf.append(o)
            if len(buf) >= 500:
                append_jsonl(os.path.join(OUT, "regulations.jsonl"), buf)
                buf = []
            if (i + 1) % 500 == 0:
                print(f"  phase2 {i+1:,}/{len(todo):,}  ({time.time()-t0:.0f}s)", flush=True)
    if buf:
        append_jsonl(os.path.join(OUT, "regulations.jsonl"), buf)
    print(f"[8-reactome] phase2 done: {len(load_jsonl(os.path.join(OUT, 'regulations.jsonl'))):,}")
    return None


# ---------------------------------------------------------------- phase 3
def phase3_expand_entities(workers=8, max_depth=8):
    """递归展开 catalyst PE 与 regulation regulator：收集 UniProt / ChEBI。
    注: ContentService 的引用字段可能是对象, 也可能是裸 dbId(int), 两者都处理。"""
    ent_path = os.path.join(OUT, "entities.jsonl")
    ents = load_jsonl(ent_path)

    reactions = load_jsonl(os.path.join(OUT, "reactions_compact.jsonl"))
    regs = load_jsonl(os.path.join(OUT, "regulations.jsonl"))
    cats = load_jsonl(os.path.join(OUT, "catalysts.jsonl"))
    seed_ids = set()
    for o in reactions.values():
        seed_ids.update(o["cat"])
    for o in cats.values():
        seed_ids.update(o["cat"])
    for o in regs.values():
        r = o.get("regulator")
        if isinstance(r, dict) and r.get("stId"):
            seed_ids.add(r["stId"])
        elif isinstance(r, int):
            seed_ids.add(r)  # 裸 dbId 也可直接 /data/query/{dbId}
    todo = [s for s in sorted(seed_ids, key=str) if str(s) not in {str(k) for k in ents}]
    print(f"[8-reactome] phase3 entities: {len(todo):,} todo / {len(ents):,} cached")

    import concurrent.futures as cf
    buf, t0 = [], time.time()

    def one(ref):
        j = fetch(ref)
        if j is None:
            return None
        keep = {k: j.get(k) for k in
                ("stId", "dbId", "schemaClass", "identifier", "referenceEntity",
                 "hasComponent", "hasMember", "component")}
        keep["stId"] = j.get("stId", str(ref))
        return keep

    # 旧缓存缺 referenceEntity 键的实体（EWAS/SimpleEntity/ChemicalDrug）→ 失效重取
    NEED_REF = {"EntityWithAccessionedSequence", "SimpleEntity", "ChemicalDrug",
                "SmallMolecule", "GenomeEncodedEntity"}
    stale = [o["stId"] for o in ents.values()
             if o.get("schemaClass") in NEED_REF and "referenceEntity" not in o]
    if stale:
        print(f"[8-reactome] phase3: invalidating {len(stale):,} stale entities")
        keep_ids = {str(o["stId"]) for o in ents.values()
                    if not (o.get("schemaClass") in NEED_REF and "referenceEntity" not in o)}
        lines = [l for l in open(ent_path)
                 if json.loads(l).get("stId") in keep_ids] if os.path.exists(ent_path) else []
        tmp = ent_path + ".tmp"
        with open(tmp, "w") as f:
            f.writelines(lines)
        os.replace(tmp, ent_path)
        ents = load_jsonl(ent_path)

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, o in enumerate(ex.map(one, todo)):
            if o is not None:
                buf.append(o)
            if len(buf) >= 500:
                append_jsonl(ent_path, buf)
                buf = []
            if (i + 1) % 500 == 0:
                print(f"  phase3 {i+1:,}/{len(todo):,}  ({time.time()-t0:.0f}s)", flush=True)
    if buf:
        append_jsonl(ent_path, buf)

    def load_with_index():
        e = load_jsonl(ent_path)
        dbidx = {o.get("dbId"): o["stId"] for o in e.values() if isinstance(o.get("dbId"), int)}
        return e, dbidx

    ents, dbidx = load_with_index()

    def resolve(ref):
        """child 引用 (stId 字符串 / 对象 / 裸 dbId int) -> ents 的 stId key 或 'NEED:<id>'。"""
        if isinstance(ref, dict):
            ref = ref.get("stId")
        if isinstance(ref, int):
            ref = dbidx.get(ref, f"NEED:{ref}")
        if isinstance(ref, str) and ref.startswith("NEED:"):
            return ref
        return ref if (ref and ref in ents) else (f"NEED:{ref}" if ref else None)

    for round_i in range(max_depth):
        need = set()

        def collect(ref, seen):
            key = resolve(ref)
            if key is None or key in seen:
                return
            seen.add(key)
            if key.startswith("NEED:"):
                need.add(key[5:])
                return
            o = ents[key]
            for k in ("hasComponent", "component", "hasMember"):
                for ch in (o.get(k) or []):
                    collect(ch, seen)

        for o in list(ents.values()):
            collect(o["stId"], set())
        if not need:
            break
        need = sorted(need, key=str)
        print(f"  phase3 round{round_i+1}: fetching {len(need):,} nested entities")
        with cf.ThreadPoolExecutor(workers) as ex:
            buf = [x for x in ex.map(one, need) if x is not None]
            append_jsonl(ent_path, buf)
        ents, dbidx = load_with_index()

    # ------------------------------------------------ 汇总边表
    def members(ref, seen=None, kind="prot"):
        """kind: 'prot'→UniProt id, 'chem'→ChEBI id（递归）。"""
        if seen is None:
            seen = set()
        key = resolve(ref)
        if key is None or key.startswith("NEED:") or key in seen:
            return set()
        seen.add(key)
        o = ents[key]
        out = set()
        cls = o.get("schemaClass")
        ident = o.get("identifier") or {}
        db = (ident.get("databaseName") or "")
        iid = ident.get("identifier")
        if kind == "prot" and cls == "EntityWithAccessionedSequence":
            # EWAS 的 UniProt 藏在 referenceEntity（dict: databaseName/identifier）
            refe = o.get("referenceEntity")
            if isinstance(refe, dict):
                rdb = (refe.get("databaseName") or "")
                rid = refe.get("identifier")
                if "uniprot" in rdb.lower() and rid:
                    out.add(rid)
        if kind == "chem" and cls in ("SimpleEntity", "ChemicalDrug", "SmallMolecule"):
            # 小分子的 ChEBI 藏在 referenceEntity（ContentService 类名是 SimpleEntity）
            refe = o.get("referenceEntity")
            if isinstance(refe, dict) and "chebi" in (refe.get("databaseName") or "").lower():
                rid = refe.get("identifier")
                if rid:
                    out.add(str(rid).split(":")[-1])
        if kind == "chem" and db == "ChEBI" and iid:
            out.add(str(iid).split(":")[-1])
        for k in ("hasComponent", "component", "hasMember"):
            for ch in (o.get(k) or []):
                out |= members(ch, seen, kind)
        return out

    rows, n_dir, n_nodir = [], 0, 0
    for reg_stid, o in regs.items():
        cls = o.get("schemaClass")
        if cls not in ("PositiveRegulation", "NegativeRegulation"):
            continue
        direction = "activation" if cls == "PositiveRegulation" else "inhibition"
        chebis = members(o.get("regulator"), kind="chem")
        if not chebis:
            n_nodir += 1
            continue
        n_dir += 1
        for rxn, r in reactions.items():
            if reg_stid in r["reg"]:
                pe_ids = cats[rxn]["cat"] if (rxn in cats and cats[rxn]["cat"]) else r["cat"]
                ups = set()
                for pe in pe_ids:
                    ups |= members(pe, kind="prot")
                if not ups:
                    continue
                for ch in sorted(chebis):
                    rows.append((ch, direction, rxn, ";".join(sorted(ups))))
                break
    out = os.path.join(OUT, "regulation_edges.tsv")
    with open(out, "w") as f:
        f.write("chebi_id\tdirection\treaction\tuniprot_set\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"[8-reactome] edges: {len(rows):,} (chemical-regulated events: {n_dir:,}, "
          f"non-chemical regulators skipped: {n_nodir:,})")
    print(f"[8-reactome] saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-phase", type=int, default=3, help="跑到第几阶段(1-3)")
    args = ap.parse_args()
    t0 = time.time()
    rxns = phase0_hierarchy()
    reg_ids, _ = phase1_reactions(rxns, args.workers)
    if args.max_phase >= 2:
        phase2_regulations(reg_ids, args.workers)
        phase2b_catalysts(args.workers)
    if args.max_phase >= 3:
        phase3_expand_entities(args.workers)
    print(f"[8-reactome] all done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
