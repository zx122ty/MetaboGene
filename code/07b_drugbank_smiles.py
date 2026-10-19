"""
Step 7b — 为 DrugBank 表回填 PubChem 结构（CID + SMILES）
============================================================
背景: data/external/drugbank/drugbank_final_result.csv 是 DrugBank 药物卡片导出
(latin-1 编码, 15,271 行), 无结构列。按 药物名 -> PubChem name 检索 回填:
  pubchem_cid / pubchem_smiles(优先 Isomeric) / pubchem_smiles_canonical /
  smile_matched_by(命中所用名称) / smile_status

失败语义: PubChem 404 = 无结构 (生物药/多肽等), 记 no_structure。
主名 404 时按 synonyms(分号分隔) 逐个回退, 每药最多试 MAX_SYNONYM_TRIALS 个。

风格沿用 code/02_get_SMILES/03_stage1a_api_fields.py: ThreadPoolExecutor +
全局令牌桶限速 (PubChem <=5 req/s, 这里 4.5) + JSON 缓存断点续跑。
(原计划放 code/02_get_SMILES/, 该目录 root 属主不可写, 故放此处。)

用法: python code/07b_drugbank_smiles.py
输出: data/external/drugbank/drugbank_final_result_with_smiles.csv
      data/external/drugbank/pubchem_name_cache.json (断点缓存)
"""

import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============ 配置 ============
DRUGBANK_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "data", "external", "drugbank"))
INPUT_CSV = os.path.join(DRUGBANK_DIR, "drugbank_final_result.csv")
OUTPUT_CSV = os.path.join(DRUGBANK_DIR, "drugbank_final_result_with_smiles.csv")
CACHE_JSON = os.path.join(DRUGBANK_DIR, "pubchem_name_cache.json")
LOG_FILE = os.path.join(DRUGBANK_DIR, "drugbank_smiles.log")

N_THREADS = 4                 # 并发线程 (PubChem 建议 <=5)
MAX_REQUESTS_PER_SEC = 4.5    # 全局限速
MAX_SYNONYM_TRIALS = 8        # 主名失败后最多回退的同义词数
SAVE_EVERY = 200              # 每 N 个名称落盘缓存
TIMEOUT = 30

PUBCHEM_COMPOUND = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"


# ============ 日志 ============
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
log = logging.getLogger(__name__)


# ============ 全局限速器 (与 03_stage1a 相同) ============
class RateLimiter:
    def __init__(self, max_per_sec):
        self.min_interval = 1.0 / max_per_sec
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            wait_s = self.min_interval - (now - self._last)
            if wait_s > 0:
                time.sleep(wait_s)
            self._last = time.monotonic()


rate_limiter = RateLimiter(MAX_REQUESTS_PER_SEC)

_thread_local = threading.local()


def get_session():
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        retries = Retry(total=3, backoff_factor=1.5,
                        status_forcelist=[429, 500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=retries))
        _thread_local.session = s
    return _thread_local.session


# ============ 缓存 ============
def load_cache():
    if os.path.exists(CACHE_JSON):
        with open(CACHE_JSON) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    tmp = CACHE_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, CACHE_JSON)


# ============ PubChem 查询 ============
def lookup_name(name):
    """名称 -> {cid, smiles, smiles_canonical} 或 None(404)。网络异常抛出。"""
    session = get_session()
    url = (f"{PUBCHEM_COMPOUND}/name/{quote(str(name), safe='')}"
           f"/property/CanonicalSMILES,IsomericSMILES/JSON")
    rate_limiter.wait()
    resp = session.get(url, timeout=TIMEOUT)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    props = (resp.json().get("PropertyTable") or {}).get("Properties") or []
    if not props:
        return None
    p = props[0]
    smiles = p.get("IsomericSMILES") or p.get("SMILES")
    canon = p.get("CanonicalSMILES") or p.get("ConnectivitySMILES")
    return {"cid": p.get("CID"), "smiles": smiles, "smiles_canonical": canon}


def resolve_drug(name, synonyms):
    """主名 + 同义词回退。返回 (info, matched_by, status)。"""
    trials = [("primary", name)]
    for s in (synonyms or [])[:MAX_SYNONYM_TRIALS]:
        s = (s or "").strip()
        if s and len(s) >= 2:
            trials.append(("synonym", s))
    for kind, nm in trials:
        try:
            info = lookup_name(nm)
        except requests.RequestException as e:
            log.warning(f"网络异常 {nm!r}: {e}")
            time.sleep(2)
            continue
        if info:
            return info, f"{kind}:{nm}", "ok"
    return None, None, "no_structure"


# ============ 主流程 ============
def main():
    df = pd.read_csv(INPUT_CSV, dtype=str, encoding="latin-1")
    log.info(f"DrugBank 行数: {len(df):,}; 唯一药物名: {df['name'].nunique():,}")

    cache = load_cache()
    names = [n for n in df["name"].dropna().unique().tolist()]
    todo = [n for n in names if n not in cache]
    log.info(f"缓存命中: {len(names) - len(todo):,}; 待查询: {len(todo):,}")

    # 每个药物名的同义词表 (分号分隔)
    syn_map = {}
    for nm, syn in zip(df["name"], df["synonyms"]):
        if nm not in syn_map:
            syn_map[nm] = [s for s in str(syn).split(";")] if pd.notna(syn) else []

    def work(nm):
        info, matched, status = resolve_drug(nm, syn_map.get(nm))
        return nm, {"cid": info["cid"] if info else None,
                    "smiles": info["smiles"] if info else None,
                    "smiles_canonical": info["smiles_canonical"] if info else None,
                    "matched_by": matched, "status": status}

    t0, done = time.time(), 0
    if todo:
        with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
            futs = {ex.submit(work, n): n for n in todo}
            for fut in as_completed(futs):
                nm, rec = fut.result()
                cache[nm] = rec
                done += 1
                if done % SAVE_EVERY == 0:
                    save_cache(cache)
                    ok = sum(1 for v in cache.values() if v.get("status") == "ok")
                    log.info(f"进度 {done:,}/{len(todo):,} ({time.time()-t0:.0f}s) "
                             f"命中结构 {ok:,}")
        save_cache(cache)

    # ---------------- 合并回表 ----------------
    df["pubchem_cid"] = df["name"].map(lambda n: cache.get(n, {}).get("cid"))
    df["pubchem_smiles"] = df["name"].map(lambda n: cache.get(n, {}).get("smiles"))
    df["pubchem_smiles_canonical"] = df["name"].map(lambda n: cache.get(n, {}).get("smiles_canonical"))
    df["smile_matched_by"] = df["name"].map(lambda n: cache.get(n, {}).get("matched_by"))
    df["smile_status"] = df["name"].map(lambda n: cache.get(n, {}).get("status"))

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    n_ok = int((df["smile_status"] == "ok").sum())
    log.info(f"完成: 命中结构 {n_ok:,}/{len(df):,} 行 -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
