#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置 B · 排序对齐（StockMixer 思想）：三阶段 MLP 排序器。

- 特征：29 维基数组 + 21 维市场状态（复用 T1 的 market_daily.csv；缺失则按 MASTER 口径自建）。
- 模型（numpy 实现，无 torch 依赖）：
    指标混合：Linear(d_in->64) -> ReLU -> Linear(64->32)
    时间混合：多尺度因果滞后 [1,5,20]（同股票内；StockMixer 多尺度 patch 的向量化近似）拼接 -> Linear(128->32) -> ReLU
    股票混合：市场状态 -> 16 维市场表征，与 32 维拼接 -> Linear(48->32) -> ReLU -> Linear(32->1)
- 损失 = MSE + pairwise 排序损失（权重 1.0:0.5，顶部可调；论文：两项都要有）。
- 训练：EV 式，只在成交行上拟合 E[return|filled]；目标 = 现行口径实现收益（已扣费）。
- 评估：与现行一致（x=0.05/+7%/-2%/0.5%费、gap-aware、Top10/前3统计），输出 IC/RankIC 与组合收益，
  同口径与 HGB 基线对照（同特征、同目标、同训练行）。
- 产出：method_combos/b_mlp_sorter/（权重 npz + 分数 npy + summary.json，全部加载校验）。

约束：只读现有数据；不修改/删除任何原始与分批文件；输出文件必须可加载。
"""
import os, sys, json, time, argparse
import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import combo_common as C

# ---------------------------------------------------------------- CONFIG
CONFIG = dict(
    # 模型结构（建议初值；论文原值：三阶段混合、MSE+pairwise）
    IND_HIDDEN=[64, 32],     # 指标混合各层维度
    TIME_LAGS=[1, 5, 20],    # 多尺度因果滞后（交易行数；StockMixer 多尺度 patch 的向量化近似）
    TIME_OUT=32,             # 时间混合输出维度
    MKT_HID=16,              # 市场表征维度
    HEAD_HID=32,             # 融合头隐层维度
    USE_GATE=False,          # D 开 True：市场引导门控（MASTER 式）
    D_EXTRA=0,               # 额外信号列数（D=2：Kronos 收益/波动率）
    # 损失
    LOSS_MSE_W=1.0,
    LOSS_PAIR_W=0.5,         # 建议初值 1.0:0.5
    PAIR_MARGIN=0.01,
    # 训练
    LR=3e-4,
    EPOCHS=3,
    BATCH_STOCKS=64,         # 每批股票数（按股票分批，保证滞后索引连续）
    PAIR_DAYS_PER_BATCH=16,  # 每批采样的日期数（pairwise 用）
    PAIR_MAX_PER_DAY=256,    # 每个日期最多采样对数
    SEED=0,
    # 冒烟
    SMOKE_N=20000,
    SMOKE_DAYS=150,          # 取最近 N 个交易日
    SMOKE_EPOCHS=2,
    SMOKE_BATCH_STOCKS=32,
)


# ---------------------------------------------------------------- 工具
def localize_lag(lag_idx_global, stock_segs, chunk_len=None):
    """把全局滞后索引映射为 batch/chunk 局部索引。
    stock_segs: [(gstart, gend, lstart), ...] 全局->局部偏移；
    无效条目（原全局索引为 0 占位）在映射后可能为负，一律置无效并裁剪到界内，
    保证 forward 中的 a2[li] 永不越界。
    """
    out = {}
    for L, (li_g, v_g) in lag_idx_global.items():
        p_li, p_v = [], []
        for gs, ge, ls in stock_segs:
            p_li.append(li_g[gs:ge] - gs + ls)
            p_v.append(v_g[gs:ge])
        li = np.concatenate(p_li)
        v = np.concatenate(p_v).copy()
        total = sum(ge - gs for gs, ge, _ in stock_segs)
        if chunk_len is not None:
            total = chunk_len
        bad = (li < 0) | (li >= total)
        v[bad] = False
        li = np.clip(li, 0, max(total - 1, 0))
        out[L] = (li.astype(np.int32), v)
    return out


def stock_segments(bounds, perm_slice):
    """由股票 bounds 生成 (gstart, gend, lstart) 列表与行索引。"""
    segs, rows, lstart = [], [], 0
    for i in perm_slice:
        s, e = bounds[i]
        segs.append((s, e, lstart))
        rows.append(np.arange(s, e)); lstart += e - s
    rows = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    return segs, rows


# ---------------------------------------------------------------- 数据准备
def prepare_data(smoke=False, smoke_n=None, splits=("tr", "va"), extra=None):
    """extra: {split: (n,k) array} 额外信号列（如 Kronos），拼到指标输入末尾。
    返回 dict(split -> dict)：Xb(29), Xm(21), Xk, days, codes, bars, realized, filled,
    order 按 (code,day) 排序后的全部数组，lag_idx, stock_bounds。
    """
    cfg = CONFIG
    out = {}
    mkt = C.ensure_market_daily()
    for split in splits:
        Xb_all, y_all = C.load_features(split)
        bars = C.load_exec_bars(split)
        days_all = bars["day"].astype(np.int64)
        n_all = len(days_all)
        assert len(Xb_all) == n_all, f"{split}: 特征行 {len(Xb_all)} != 价格行 {n_all}"
        if smoke:
            ud = np.unique(days_all)
            keep_days = set(ud[np.argsort(ud)[-cfg["SMOKE_DAYS"]:]].tolist())
            mask = np.array([d in keep_days for d in days_all])
            idx = np.where(mask)[0][: (smoke_n or cfg["SMOKE_N"])]
        else:
            idx = np.arange(n_all)
        Xb = np.asarray(Xb_all[idx], dtype=np.float32)
        days = days_all[idx]
        codes = bars["code"][idx].astype(np.int64)
        bsub = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in bars.items()
                if k in ("sig", "bl", "sl", "sh", "sc", "bo", "so", "day", "code")}
        bsub["has_open"] = bars["has_open"]; bsub["codes"] = bars["codes"]
        Xm, mkt_cols = C.market_matrix(days.astype(np.int32), mkt)
        filled, realized = C.realized_current_cfg(bsub)
        Xk = None
        if extra is not None and split in extra:
            Xk = np.asarray(extra[split], dtype=np.float32)
            assert len(Xk) == len(idx), "extra 信号行数不匹配"
        # (code, day) 排序，保证滞后索引连续
        order = np.lexsort((days, codes))
        S = lambda a: a[order]
        sd, sc_ = days[order], codes[order]
        lag_idx = {}
        ar = np.arange(len(order))
        for L in cfg["TIME_LAGS"]:
            prev = ar - L
            ok = prev >= 0
            same = np.zeros(len(order), dtype=bool)
            dgap = np.zeros(len(order), dtype=np.int32)
            same[ok] = sc_[prev[ok]] == sc_[ok]
            dgap[ok] = (sd[ok] - sd[prev[ok]]).astype(np.int32)
            valid = ok & same & (dgap > 0) & (dgap <= L + 2)
            lag_idx[L] = (np.where(valid, prev, 0).astype(np.int32), valid)
        uc, starts = np.unique(sc_, return_index=True)
        ends = np.append(starts[1:], len(order))
        out[split] = {
            "Xb": S(Xb), "Xm": S(Xm), "Xk": S(Xk) if Xk is not None else None,
            "days": sd, "codes": sc_, "bars": {k: (S(v) if isinstance(v, np.ndarray) else v)
                                               for k, v in bsub.items()},
            "realized": S(realized), "filled": S(filled),
            "lag_idx": lag_idx, "stock_bounds": list(zip(starts.tolist(), ends.tolist())),
            "mkt_cols": mkt_cols, "n_all": n_all, "n_use": len(order),
        }
        print(f"[b] {split}: 使用 {len(order)}/{n_all} 行, {len(uc)} 只股票, "
              f"成交率 {filled.mean():.3f}", flush=True)
    # 行序抽查（mkt_ret 校验）
    chk = C.spot_check_alignment(out["tr"]["Xb"][:5000], out["tr"]["days"][:5000].astype(np.int32))
    out["_align_check"] = chk
    return out


# ---------------------------------------------------------------- numpy MLP
def _relu(x): return np.maximum(x, 0)
def _drelu(x): return (x > 0).astype(np.float32)
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class MixMLP:
    """三阶段混合 MLP（numpy）。d_in = 29(+gate) + 21 + d_extra。"""

    def __init__(self, d_extra=0, use_gate=False, seed=0):
        cfg = CONFIG
        self.d_extra, self.use_gate = d_extra, use_gate
        rng = np.random.default_rng(seed)
        d_ind = 29 + 21 + d_extra
        h1, h2 = cfg["IND_HIDDEN"]
        tl = len(cfg["TIME_LAGS"])
        self.p = {}
        def init(fan_in, fan_out):
            return (rng.standard_normal((fan_in, fan_out)).astype(np.float32)
                    * np.sqrt(2.0 / fan_in))
        self.p["W1"] = init(d_ind, h1); self.p["b1"] = np.zeros(h1, np.float32)
        self.p["W2"] = init(h1, h2);    self.p["b2"] = np.zeros(h2, np.float32)
        self.p["W3"] = init(h2 * (1 + tl), cfg["TIME_OUT"]); self.p["b3"] = np.zeros(cfg["TIME_OUT"], np.float32)
        self.p["Wm"] = init(21, cfg["MKT_HID"]);             self.p["bm"] = np.zeros(cfg["MKT_HID"], np.float32)
        self.p["W4"] = init(cfg["TIME_OUT"] + cfg["MKT_HID"], cfg["HEAD_HID"])
        self.p["b4"] = np.zeros(cfg["HEAD_HID"], np.float32)
        self.p["W5"] = init(cfg["HEAD_HID"], 1);             self.p["b5"] = np.zeros(1, np.float32)
        if use_gate:
            self.p["Wg"] = init(21, 29) * 0.1; self.p["bg"] = np.zeros(29, np.float32)
        # Adam 状态
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, Xb, Xm, Xk, lag_idx):
        """Xb(n,29) Xm(n,21)；返回 scores(n,) 与 cache。"""
        cfg = CONFIG
        if self.use_gate:
            g = _sigmoid(Xm @ self.p["Wg"] + self.p["bg"])   # MASTER 式市场引导门控
            Xb_in = Xb * g
        else:
            Xb_in = Xb
        parts = [Xb_in, Xm] + ([Xk] if Xk is not None else [])
        X = np.concatenate(parts, axis=1).astype(np.float32)
        h1 = X @ self.p["W1"] + self.p["b1"]; a1 = _relu(h1)
        h2 = a1 @ self.p["W2"] + self.p["b2"]; a2 = _relu(h2)
        lags = []
        lag_vals = []
        for L in cfg["TIME_LAGS"]:
            li, valid = lag_idx[L]
            el = a2[li] * valid[:, None].astype(np.float32)
            lags.append((li, valid))   # 反向散射用
            lag_vals.append(el)        # 前向拼接用
        F = np.concatenate([a2] + lag_vals, axis=1)
        h3 = F @ self.p["W3"] + self.p["b3"]; a3 = _relu(h3)
        hm = Xm @ self.p["Wm"] + self.p["bm"]; am = _relu(hm)
        H = np.concatenate([a3, am], axis=1)
        h4 = H @ self.p["W4"] + self.p["b4"]; a4 = _relu(h4)
        s = (a4 @ self.p["W5"] + self.p["b5"]).ravel()
        cache = dict(X=X, h1=h1, a1=a1, h2=h2, a2=a2, lags=lags, F=F, h3=h3, a3=a3,
                     hm=hm, am=am, H=H, h4=h4, a4=a4, Xb=Xb, Xm=Xm, g=(g if self.use_gate else None))
        return s, cache

    def loss_and_grad(self, s, cache, r, filled, days, rng):
        cfg = CONFIG
        m = filled
        n_m = max(m.sum(), 1)
        # MSE（只在成交行）
        diff = s[m] - r[m]
        mse = float((diff ** 2).mean()) if m.any() else 0.0
        ds = np.zeros_like(s)
        if m.any():
            ds[m] = 2.0 * diff / n_m * cfg["LOSS_MSE_W"]
        # pairwise（同日期内采样）
        pair = 0.0
        if cfg["LOSS_PAIR_W"] > 0 and m.sum() >= 4:
            ud = np.unique(days[m])
            sel_days = ud[rng.choice(len(ud), size=min(cfg["PAIR_DAYS_PER_BATCH"], len(ud)), replace=False)]
            tot = 0.0; cnt = 0
            for d in sel_days:
                idx = np.where(m & (days == d))[0]
                if len(idx) < 2: continue
                P = min(cfg["PAIR_MAX_PER_DAY"], len(idx) * (len(idx) - 1) // 2)
                ii = idx[rng.integers(0, len(idx), P)]; jj = idx[rng.integers(0, len(idx), P)]
                keep = r[ii] > r[jj]
                ii, jj = ii[keep], jj[keep]
                if len(ii) == 0: continue
                viol = cfg["PAIR_MARGIN"] - (s[ii] - s[jj])
                act = viol > 0
                tot += float(np.maximum(viol, 0).mean())
                w = cfg["LOSS_PAIR_W"] / max(act.sum(), 1)
                np.add.at(ds, ii[act], -w)   # 必须用 add.at，重复索引才能累加
                np.add.at(ds, jj[act], w)
                cnt += 1
            pair = tot / max(cnt, 1)
        total = cfg["LOSS_MSE_W"] * mse + cfg["LOSS_PAIR_W"] * pair
        grads = self._backward(ds, cache)
        return total, mse, pair, grads

    def _backward(self, ds, cache):
        p = self.p; g = {}
        # head
        dh5 = ds[:, None]                                   # (n,1)
        g["W5"] = cache["a4"].T @ dh5; g["b5"] = dh5.sum(0)
        da4 = (dh5 @ p["W5"].T) * _drelu(cache["h4"])
        g["W4"] = cache["H"].T @ da4; g["b4"] = da4.sum(0)
        dH = da4 @ p["W4"].T
        d_a3, d_am = dH[:, :p["W3"].shape[1]], dH[:, p["W3"].shape[1]:]
        # market branch
        dhm = d_am * _drelu(cache["hm"])
        g["Wm"] = cache["Xm"].T @ dhm; g["bm"] = dhm.sum(0)
        dXm_from_m = dhm @ p["Wm"].T
        # time mix
        dh3 = d_a3 * _drelu(cache["h3"])
        g["W3"] = cache["F"].T @ dh3; g["b3"] = dh3.sum(0)
        dF = dh3 @ p["W3"].T
        h2d = p["W2"].shape[1]
        da2 = dF[:, :h2d].copy()   # copy：后面用 add.at 做散射累加
        off = h2d
        for li, valid in cache["lags"]:   # 与 forward 中 TIME_LAGS 顺序一致
            dl = dF[:, off:off + h2d]; off += h2d
            # 反向通过滞后 gather：梯度散射回原位置
            np.add.at(da2, li[valid], dl[valid])
        dh2 = da2 * _drelu(cache["h2"])
        g["W2"] = cache["a1"].T @ dh2; g["b2"] = dh2.sum(0)
        da1 = (dh2 @ p["W2"].T) * _drelu(cache["h1"])
        g["W1"] = cache["X"].T @ da1; g["b1"] = da1.sum(0)
        dX = da1 @ p["W1"].T
        # gate 分支
        dXm = dXm_from_m
        if self.use_gate:
            dXb_in = dX[:, :29]; dXm_extra = dX[:, 29:50]
            dXm = dXm + dXm_extra
            Xb, Xm_ = cache["Xb"], cache["Xm"]; gg = cache["g"]
            dg = dXb_in * Xb * gg * (1 - gg)
            g["Wg"] = Xm_.T @ dg; g["bg"] = dg.sum(0)
            # dXb 未继续回传（输入层）
        return g

    def adam_step(self, grads, lr):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self.p:
            gv = grads[k].astype(np.float32)
            # 梯度裁剪防爆
            nrm = np.linalg.norm(gv)
            if nrm > 5.0: gv = gv * (5.0 / nrm)
            self.m[k] = b1 * self.m[k] + (1 - b1) * gv
            self.v[k] = b2 * self.v[k] + (1 - b2) * gv * gv
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)

    def save(self, path):
        np.savez_compressed(path, **{k: v for k, v in self.p.items()},
                            d_extra=np.array(self.d_extra), use_gate=np.array(self.use_gate))

    @staticmethod
    def load(path):
        z = np.load(path)
        d_extra, use_gate = int(z["d_extra"]), bool(z["use_gate"])
        mlp = MixMLP(d_extra=d_extra, use_gate=use_gate)
        for k in mlp.p: mlp.p[k] = z[k]
        z.close()
        return mlp


# ---------------------------------------------------------------- 训练
def train_model(data, d_extra=0, use_gate=False, epochs=None, smoke=False, tag="mlp"):
    cfg = CONFIG
    tr, va = data["tr"], data["va"]
    mlp = MixMLP(d_extra=d_extra, use_gate=use_gate, seed=cfg["SEED"])
    rng = np.random.default_rng(cfg["SEED"])
    epochs = cfg["SMOKE_EPOCHS"] if smoke else (epochs or cfg["EPOCHS"])
    bs = cfg["SMOKE_BATCH_STOCKS"] if smoke else cfg["BATCH_STOCKS"]
    nb = len(tr["stock_bounds"])
    hist = []
    for ep in range(epochs):
        t0 = time.time()
        perm = rng.permutation(nb)
        tot_loss, tot_mse, nbatches = 0.0, 0.0, 0
        for bi in range(0, nb, bs):
            segs, rows = stock_segments(tr["stock_bounds"], perm[bi:bi + bs])
            if len(rows) < 8: continue
            lag_b = localize_lag(tr["lag_idx"], segs)
            s, cache = mlp.forward(tr["Xb"][rows], tr["Xm"][rows],
                                   tr["Xk"][rows] if tr["Xk"] is not None else None,
                                   lag_b)
            total, mse, pair, grads = mlp.loss_and_grad(
                s, cache, tr["realized"][rows], tr["filled"][rows], tr["days"][rows], rng)
            if not np.isfinite(total):
                print(f"[{tag}] ep{ep} batch{bi}: loss 非有限，跳过", flush=True); continue
            mlp.adam_step(grads, cfg["LR"])
            tot_loss += total; tot_mse += mse; nbatches += 1
        # 验证
        vs = predict_all(mlp, va)
        vm = va["filled"].mean()
        vloss = float((((vs[va["filled"]] - va["realized"][va["filled"]]) ** 2).mean())) if va["filled"].any() else float("nan")
        met, _ = C.evaluate(vs, va["bars"], tag=f"{tag}-val-ep{ep}", verbose=False)
        hist.append({"epoch": ep, "train_loss": tot_loss / max(nbatches, 1),
                     "val_mse": vloss, "val_IC": met["IC"], "val_top3": met["top3_mean"]})
        ic_s = f"{met['IC']:+.4f}" if met["IC"] is not None else "n/a"
        tp3_s = f"{met['top3_mean']:+.4%}" if met["top3_mean"] is not None else "n/a"
        print(f"[{tag}] ep{ep}: train_loss={tot_loss/max(nbatches,1):.4f} "
              f"val_mse={vloss:.4f} val_IC={ic_s} val_top3={tp3_s} "
              f"{time.time()-t0:.0f}s", flush=True)
    return mlp, hist


def predict_all(mlp, split_data, chunk=200000):
    n = len(split_data["days"])
    out = np.empty(n, dtype=np.float32)
    bounds = split_data["stock_bounds"]
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        segs = []
        for (s, e) in bounds:
            if e <= i: continue
            if s >= j: break
            segs.append((max(s, i), min(e, j), max(s, i) - i))
        lag_c = localize_lag(split_data["lag_idx"], segs, chunk_len=j - i)
        r = np.arange(i, j)
        s, _ = mlp.forward(split_data["Xb"][r], split_data["Xm"][r],
                           split_data["Xk"][r] if split_data["Xk"] is not None else None,
                           lag_c)
        out[r] = s
    return out


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="配置B: 三阶段MLP排序器 (MSE+pairwise)")
    ap.add_argument("--smoke", action="store_true", help="小样本冒烟：端到端跑通")
    ap.add_argument("--smoke_n", type=int, default=CONFIG["SMOKE_N"])
    ap.add_argument("--out", default=os.path.join(C.COMBO_DIR, "b_mlp_sorter"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no_pair", action="store_true", help="关闭 pairwise 损失（消融）")
    args = ap.parse_args()
    if args.no_pair: CONFIG["LOSS_PAIR_W"] = 0.0
    t0 = time.time()
    print(f"[b] 启动 (smoke={args.smoke})", flush=True)

    data = prepare_data(smoke=args.smoke, smoke_n=args.smoke_n)
    mlp, hist = train_model(data, smoke=args.smoke, epochs=args.epochs, tag="b-mlp")

    results = {}
    all_arrays = {}
    hgb = None
    for split in ("tr", "va"):
        d = data[split]
        scores = predict_all(mlp, d)
        met, arr = C.evaluate(scores, d["bars"], tag=f"b-mlp-{split}")
        results[f"mlp_{split}"] = met
        # HGB 基线：同特征、同目标、同训练行
        Xind = np.concatenate([d["Xb"], d["Xm"]], axis=1)
        if split == "tr":
            hgb = C.train_hgb(Xind, d["realized"], d["filled"], tiny=args.smoke)
        hs = hgb.predict(Xind).astype(np.float32)
        hmet, harr = C.evaluate(hs, d["bars"], tag=f"b-hgb-{split}")
        results[f"hgb_{split}"] = hmet
        all_arrays[f"{split}_mlp_scores"] = scores
        all_arrays[f"{split}_hgb_scores"] = hs
        for k, v in arr.items(): all_arrays[f"{split}_mlp_{k}"] = v
        for k, v in harr.items(): all_arrays[f"{split}_hgb_{k}"] = v

    os.makedirs(args.out, exist_ok=True)
    mlp.save(os.path.join(args.out, "mlp_weights.npz"))
    MixMLP.load(os.path.join(args.out, "mlp_weights.npz"))  # 加载校验
    summary = {
        "config_name": "B-排序对齐 (StockMixer思想: 三阶段MLP + MSE/pairwise)",
        "CONFIG": {k: v for k, v in CONFIG.items()},
        "data": {s: {"n_use": data[s]["n_use"], "n_all": data[s]["n_all"]} for s in ("tr", "va")},
        "align_check": data["_align_check"],
        "market_cols": data["tr"]["mkt_cols"],
        "train_history": hist,
        "results": results,
        "smoke": bool(args.smoke),
        "elapsed_s": round(time.time() - t0, 1),
        "note": "冒烟模式为小样本流程验证；全量训练按 T1->T6 顺序另行安排，结果数字以全量为准",
    }
    C.save_summary(args.out, summary, arrays=all_arrays, model=hgb, model_name="hgb_baseline.pkl")
    print(f"[b] DONE {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
