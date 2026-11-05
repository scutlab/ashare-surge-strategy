#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置 C · 信号融合（Kronos 加入）。

- 先尝试下载 Kronos 预训练权重（github.com/shiyu-coder/Kronos）；
  若网络不通：冒烟模式用明确标注的占位信号跑通流程（kronos_real=false，大写告警），
  非冒烟模式直接报错退出（中文提示），绝不静默跳过。
- 2 列信号（预测收益、预测波动率）并入特征矩阵（默认 29+2，对照最干净）。
- 先跑 HGB；可选置信度加权：波动率最高 3 只降权 50%（默认开）。
- 行序按 (code, day) 排序，与 b/d 脚本一致，保证跨脚本信号可对齐。
- 产出：method_combos/c_kronos_fusion/（信号 npy + 模型 pkl + summary.json，全部加载校验）。
约束：只读现有数据（权重下载除外）；不修改/删除任何原始与分批文件。
"""
import os
import sys
import json
import time
import argparse
import subprocess
import shutil
import numpy as np
import pandas as pd

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import combo_common as C

CONFIG = dict(
    KRONOS_DIR=os.path.join(C.COMBO_DIR, "kronos_repo"),
    CLONE_URL="https://github.com/shiyu-coder/Kronos",
    CLONE_TIMEOUT_S=120,
    # 权重在 HuggingFace Hub（git 仓库只含代码）。全量推理另需: pip install torch huggingface_hub
    HF_TOKENIZER="NeoQuasar/Kronos-Tokenizer-base",
    HF_MODEL="NeoQuasar/Kronos-small",   # small/base/large 可选
    # 本地权重目录（优先于 Hub）：weights/kronos-small 与 weights/kronos-tokenizer-base
    WEIGHTS_DIR=os.path.join(C.COMBO_DIR, "c_kronos_fusion", "weights"),
    PRED_LEN=5,          # 向前预测天数
    SAMPLE_COUNT=8,      # 概率采样路径数（用于估计波动率）
    ADD_MKT=False,        # True 则特征为 29+21+2；默认 False：只加 2 列信号，对照最干净
    CONF_WEIGHT=True,     # 置信度加权开关（默认开）：波动率最高 K 只降权
    CONF_TOPK=3,          # 建议初值
    CONF_FACTOR=0.5,      # 建议初值：降权 50%
    SEQ_LEN=512,          # Kronos 输入 K 线长度（small/base 的 max_context）
    INFER_STRIDE=20,      # 全量模式每只股票推理步长（天）
    SMOKE_N=20000,
    SMOKE_DAYS=150,
)


class KronosUnavailable(Exception):
    pass


class KronosAPIError(Exception):
    pass


def ensure_kronos(smoke=False):
    cfg = CONFIG
    kd = cfg["KRONOS_DIR"]
    err = None
    if os.path.isdir(kd) and os.listdir(kd):
        try:
            return KronosWrapper(kd), True
        except Exception as e:
            err = "local load failed: %s: %s" % (type(e).__name__, e)
    else:
        try:
            print("[c] 正在下载 Kronos 权重: %s ..." % cfg["CLONE_URL"], flush=True)
            if os.path.isdir(kd):
                shutil.rmtree(kd)
            subprocess.run(
                ["git", "clone", "--depth", "1", cfg["CLONE_URL"], kd],
                timeout=cfg["CLONE_TIMEOUT_S"], check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("[c] 下载成功，尝试加载模型…", flush=True)
            return KronosWrapper(kd), True
        except Exception as e:
            err = "下载失败: %s: %s" % (type(e).__name__, e)
    if smoke:
        print("=" * 70, flush=True)
        print("[c] ！！冒烟占位警告：Kronos 权重不可用，本次用确定性占位信号跑流程，", flush=True)
        print("[c] ！！所有指标仅为流程验证，不代表 Kronos 真实效果（summary.json kronos_real=false）。", flush=True)
        print("=" * 70, flush=True)
        return StubKronos(), False
    print("[c] Kronos 不可用（%s）" % err, file=sys.stderr, flush=True)
    print("[c] 请检查网络后手动执行: git clone --depth 1 %s %s"
          % (cfg["CLONE_URL"], kd), file=sys.stderr, flush=True)
    print("【错误】Kronos 预训练权重无法获取，配置 C 需要真实权重才能跑。"
          "请按上方提示手动下载后重试。本次未产生任何结果文件。",
          file=sys.stderr, flush=True)
    sys.exit(2)


def _local_or_hub(local_dir, hub_id):
    """本地 weights/ 目录有权重文件则优先用本地，否则回退到 Hub ID。"""
    try:
        if os.path.isdir(local_dir) and any(
                f.endswith((".safetensors", ".bin", ".pt")) for f in os.listdir(local_dir)):
            return local_dir
    except OSError:
        pass
    return hub_id


class KronosWrapper:
    """真实 Kronos 包装（按仓库 README 的实际 API；权重来自本地 weights/ 或 HuggingFace Hub）。

    需要环境：pip install torch huggingface_hub。git 仓库只含代码，
    权重优先用 WEIGHTS_DIR 下的本地快照，否则从 HF Hub 下载（NeoQuasar/Kronos-small 等）。
    """

    def __init__(self, repo_dir):
        self.repo_dir = repo_dir
        cfg = CONFIG
        sys.path.insert(0, repo_dir)
        try:
            from model import Kronos, KronosTokenizer, KronosPredictor
        except Exception as e:
            raise KronosAPIError(
                "导入 Kronos 代码失败（%s: %s）。请确认 %s 为完整克隆且含 model/ 目录。"
                % (type(e).__name__, e, repo_dir))
        try:
            tok_src = _local_or_hub(os.path.join(cfg["WEIGHTS_DIR"], "kronos-tokenizer-base"),
                                    cfg["HF_TOKENIZER"])
            mdl_src = _local_or_hub(os.path.join(cfg["WEIGHTS_DIR"], "kronos-small"),
                                    cfg["HF_MODEL"])
            print("[c] Kronos 权重来源: tokenizer=%s model=%s" % (tok_src, mdl_src), flush=True)
            self.tokenizer = KronosTokenizer.from_pretrained(tok_src)
            self.model = Kronos.from_pretrained(mdl_src)
            self.predictor = KronosPredictor(self.model, self.tokenizer, max_context=512)
        except Exception as e:
            raise KronosAPIError(
                "Kronos 权重/依赖加载失败（%s: %s）。"
                "请先执行 pip install torch huggingface_hub，并确认能访问 HuggingFace Hub。"
                % (type(e).__name__, e))

    def predict(self, ohlcv):
        """ohlcv: (T,6) [open,high,low,close,volume,amount]；返回 (pred_ret, pred_vol)。
        用未来 PRED_LEN 天预测收盘均值算收益、采样路径标准差算波动率。"""
        cfg = CONFIG
        try:
            n = len(ohlcv)
            df = pd.DataFrame(ohlcv, columns=["open", "high", "low", "close", "volume", "amount"])
            x_ts = pd.Series(pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B"))
            y_ts = pd.Series(pd.bdate_range(start=x_ts.iloc[-1] + pd.offsets.BDay(1),
                                            periods=cfg["PRED_LEN"]))
            pred_df = self.predictor.predict(
                df=df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=cfg["PRED_LEN"], T=1.0, top_p=0.9,
                sample_count=cfg["SAMPLE_COUNT"])
            closes = np.asarray(pred_df["close"], dtype=np.float64).ravel()
            last = float(ohlcv[-1, 3])
            if last <= 0 or len(closes) == 0:
                return 0.0, 0.02
            pred_ret = float(closes.mean() / last - 1)
            pred_vol = float(closes.std() / last) if len(closes) > 1 else 0.02
            return pred_ret, max(pred_vol, 1e-4)
        except KronosAPIError:
            raise
        except Exception as e:
            raise KronosAPIError("Kronos 推理失败（%s: %s），请对照仓库 README 检查。"
                                 % (type(e).__name__, e))


class StubKronos:
    def predict(self, ohlcv):
        return 0.0, 0.02

    def pseudo_for(self, n):
        rng = np.random.default_rng(12345)
        pr = rng.uniform(-0.02, 0.02, size=n).astype(np.float32)
        pv = rng.uniform(0.01, 0.05, size=n).astype(np.float32)
        return np.stack([pr, pv], axis=1)


def build_signals(splits, smoke, smoke_n, kronos, is_real, data_days_codes):
    sig = {}
    for split in splits:
        n = len(data_days_codes[split][0])
        if is_real and not smoke:
            sig[split] = build_signals_full(split, data_days_codes[split], kronos)
        elif isinstance(kronos, StubKronos):
            sig[split] = kronos.pseudo_for(n)
        else:
            rng = np.random.default_rng(987)
            pr = rng.uniform(-0.02, 0.02, size=n).astype(np.float32)
            pv = rng.uniform(0.01, 0.05, size=n).astype(np.float32)
            sig[split] = np.stack([pr, pv], axis=1)
    return sig


def build_signals_full(split, days_codes, kronos):
    cfg = CONFIG
    days, codes = days_codes
    raw_files = [os.path.join(C.BASE, f) for f in ("full_raw_partial.pkl", "full_new_partial.pkl")]
    raw_files = [f for f in raw_files if os.path.exists(f)]
    if not raw_files:
        raise FileNotFoundError("原始行情文件缺失，无法做 Kronos 全量推理")
    code_names = json.load(open(os.path.join(C.PANEL_DIR, "codes.json")))["codes"]
    name_of = np.array(code_names)
    last_sig = {}
    for rf in raw_files:
        d = pd.read_pickle(rf)
        d = d[d["code"].isin(set(code_names))].sort_values(["code", "date"])
        for code, g in d.groupby("code", sort=False):
            if len(g) < 60:
                continue
            ohlcv = g[["open", "high", "low", "close", "volume"]].to_numpy(np.float64)
            amt = (g["close"] * g["volume"]).to_numpy(np.float64)
            seq = np.stack([ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3],
                            ohlcv[:, 4], amt], axis=1)
            pr, pv = kronos.predict(seq[-cfg["SEQ_LEN"]:])
            last_sig[code] = (pr, pv)
        del d
    sig = np.zeros((len(days), 2), dtype=np.float32)
    sig[:, 1] = 0.02
    for i, cd in enumerate(codes):
        code = name_of[cd]
        if code in last_sig:
            sig[i] = last_sig[code]
    return sig


def run(splits=("tr", "va"), smoke=False, smoke_n=None, out=None,
        kronos=None, is_real=None):
    cfg = CONFIG
    out = out or os.path.join(C.COMBO_DIR, "c_kronos_fusion")
    t0 = time.time()
    if kronos is None:
        kronos, is_real = ensure_kronos(smoke=smoke)

    data = {}
    mkt = C.ensure_market_daily() if cfg["ADD_MKT"] else None
    for split in splits:
        Xb_all, _ = C.load_features(split)
        bars = C.load_exec_bars(split)
        days_all = bars["day"].astype(np.int64)
        if smoke:
            ud = np.unique(days_all)
            keep = set(ud[np.argsort(ud)[-cfg["SMOKE_DAYS"]:]].tolist())
            idx = np.where([d in keep for d in days_all])[0][: (smoke_n or cfg["SMOKE_N"])]
        else:
            idx = np.arange(len(days_all))
        Xb = np.asarray(Xb_all[idx], dtype=np.float32)
        days = days_all[idx]
        codes = bars["code"][idx].astype(np.int64)
        bsub = {}
        for k, v in bars.items():
            if k in ("sig", "bl", "sl", "sh", "sc", "bo", "so", "day", "code"):
                bsub[k] = v[idx] if isinstance(v, np.ndarray) else v
        bsub["has_open"] = bars["has_open"]
        bsub["codes"] = bars["codes"]
        filled, realized = C.realized_current_cfg(bsub)
        order = np.lexsort((days, codes))
        Xb = Xb[order]
        days = days[order]
        codes = codes[order]
        filled = filled[order]
        realized = realized[order]
        bsub2 = {}
        for k, v in bsub.items():
            bsub2[k] = v[order] if isinstance(v, np.ndarray) else v
        bsub = bsub2
        feats = [Xb]
        feat_names = ["base29"]
        if mkt is not None:
            Xm, _ = C.market_matrix(days.astype(np.int32), mkt)
            feats.append(Xm)
            feat_names.append("mkt21")
        data[split] = {"Xb": Xb, "days": days, "codes": codes, "bars": bsub,
                       "filled": filled, "realized": realized,
                       "feat_names": feat_names, "_feats_extra": feats[1:]}

    sig = build_signals(splits, smoke, smoke_n, kronos, is_real,
                        {s: (data[s]["days"], data[s]["codes"]) for s in splits})

    results = {}
    all_arrays = {}
    hgb = None
    for split in splits:
        d = data[split]
        X = np.concatenate([d["Xb"]] + d["_feats_extra"] + [sig[split]], axis=1)
        if split == "tr":
            hgb = C.train_hgb(X, d["realized"], d["filled"], tiny=smoke)
        hs = hgb.predict(X).astype(np.float32)
        kv = sig[split][:, 1]
        dys = d["days"]

        def adjust(s, b, _kv=kv, _dys=dys):
            if not cfg["CONF_WEIGHT"]:
                return s
            s = s.copy()
            for dd in np.unique(_dys):
                m = _dys == dd
                idx2 = np.where(m)[0]
                if len(idx2) <= cfg["CONF_TOPK"]:
                    continue
                topk = idx2[np.argsort(-_kv[idx2])[:cfg["CONF_TOPK"]]]
                s[topk] *= cfg["CONF_FACTOR"]
            return s

        met, arr = C.evaluate_with_adjusted_scores(hs, d["bars"], adjust_fn=adjust,
                                                   tag="c-hgb-%s" % split)
        results[split] = met
        all_arrays["%s_signals" % split] = sig[split]
        all_arrays["%s_scores" % split] = hs
        all_arrays["%s_days" % split] = d["days"]
        all_arrays["%s_codes" % split] = d["codes"]
        for k, v in arr.items():
            all_arrays["%s_%s" % (split, k)] = v

    summary = {
        "config_name": "C-signal-fusion (Kronos 2-col signals + HGB)",
        "CONFIG": {k: v for k, v in CONFIG.items()},
        "kronos_real": bool(is_real),
        "kronos_repo": cfg["KRONOS_DIR"],
        "feature_cols": ["base29"] + (["mkt21"] if cfg["ADD_MKT"] else []) + ["kronos_ret", "kronos_vol"],
        "results": results,
        "smoke": bool(smoke),
        "elapsed_s": round(time.time() - t0, 1),
        "warning": None if is_real else "占位信号！指标仅为流程验证，不代表 Kronos 真实效果",
    }
    paths = C.save_summary(out, summary, arrays=all_arrays, model=hgb,
                           model_name="hgb_kronos.pkl")
    print("[c] DONE %.0fs -> %s (kronos_real=%s)" % (time.time() - t0, out, is_real), flush=True)
    return summary, out, paths


def main():
    ap = argparse.ArgumentParser(description="Config C: Kronos signal fusion")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_n", type=int, default=CONFIG["SMOKE_N"])
    ap.add_argument("--out", default=os.path.join(C.COMBO_DIR, "c_kronos_fusion"))
    ap.add_argument("--no_conf_weight", action="store_true")
    args = ap.parse_args()
    if args.no_conf_weight:
        CONFIG["CONF_WEIGHT"] = False
    run(smoke=args.smoke, smoke_n=args.smoke_n, out=args.out)


if __name__ == "__main__":
    main()
