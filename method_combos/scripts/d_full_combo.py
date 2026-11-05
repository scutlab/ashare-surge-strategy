#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置 D · 完全体（B 的 MLP + A 的门控 + C 的信号，三合一）。

- 模块化：默认优先消费 c 脚本的产出文件（arrays.npz 中的 {split}_signals）；
  加 --recompute 可独立重算（import c 模块在进程内跑，不读产出文件）。
- 模型：b 的 MixMLP，d_extra=2（Kronos 收益/波动率），use_gate=True
  （MASTER 式市场引导门控：sigmoid(Xm@Wg+bg) 作用于 29 维基特征）。
- 损失/训练/评估与 b 一致（MSE + pairwise；现行口径 x=0.05/+7%/-2%/0.5%费、gap-aware、Top10/前3统计）。
- 产出：method_combos/d_full_combo/（权重 npz + 分数 npy + summary.json，全部加载校验）。

约束：只读现有数据与 b/c 产出；不修改/删除任何原始与分批文件。
"""
import os
import sys
import json
import time
import argparse
import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import combo_common as C
import b_mlp_sorter as B
import c_kronos_fusion as CF

# ---------------------------------------------------------------- CONFIG
CONFIG = dict(
    C_OUT=os.path.join(C.COMBO_DIR, "c_kronos_fusion"),
    SMOKE_N=20000,
    SMOKE_DAYS=150,
)


def load_c_signals(c_out, splits, n_use):
    """消费 c 脚本产出：arrays.npz 中的 {split}_signals/{split}_days/{split}_codes。
    缺失或行数/键不一致则中文明示退出（提示先跑 c 或加 --recompute）。"""
    ap = os.path.join(c_out, "arrays.npz")
    sp = os.path.join(c_out, "summary.json")
    if not (os.path.exists(ap) and os.path.exists(sp)):
        print("【错误】c 产出缺失：%s 下没有 arrays.npz/summary.json。" % c_out,
              file=sys.stderr, flush=True)
        print("请先跑: python c_kronos_fusion.py --smoke   或本次加 --recompute 独立重算。",
              file=sys.stderr, flush=True)
        sys.exit(2)
    summ = json.load(open(sp))
    z = np.load(ap)
    sig = {}
    try:
        for s in splits:
            k = "%s_signals" % s
            if k not in z.files:
                print("【错误】c 产出中缺少 %s。请重跑 c 或加 --recompute。" % k,
                      file=sys.stderr, flush=True)
                sys.exit(2)
            v = z[k]
            if len(v) != n_use[s]:
                print("【错误】c 信号行数(%d)与 d 数据行数(%d, split=%s)不一致，"
                      "请加 --recompute 独立重算。" % (len(v), n_use[s], s),
                      file=sys.stderr, flush=True)
                sys.exit(2)
            sig[s] = v.astype(np.float32)
    finally:
        z.close()
    print("[d] 消费 c 产出：kronos_real=%s, smoke=%s"
          % (summ.get("kronos_real"), summ.get("smoke")), flush=True)
    return sig, summ


def main():
    ap = argparse.ArgumentParser(description="配置D: 三合一完全体")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_n", type=int, default=CONFIG["SMOKE_N"])
    ap.add_argument("--out", default=os.path.join(C.COMBO_DIR, "d_full_combo"))
    ap.add_argument("--recompute", action="store_true",
                    help="独立重算：import c 模块在进程内重跑信号，不消费产出文件")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    print("[d] 启动 (smoke=%s, recompute=%s)" % (args.smoke, args.recompute), flush=True)

    splits = ("tr", "va")
    kronos_real, kronos_warn = None, None
    # 先准备基础数据（确定性子集），再决定信号来源
    data = B.prepare_data(smoke=args.smoke, smoke_n=args.smoke_n, splits=splits)
    n_use = {s: data[s]["n_use"] for s in splits}
    if args.recompute:
        kronos, is_real = CF.ensure_kronos(smoke=args.smoke)
        kronos_real = bool(is_real)
        sig = CF.build_signals(
            splits, args.smoke, args.smoke_n, kronos, is_real,
            {s: (data[s]["days"], data[s]["codes"]) for s in splits})
        if not is_real:
            kronos_warn = "占位信号！指标仅为流程验证，不代表 Kronos 真实效果"
    else:
        sig, c_summ = load_c_signals(CONFIG["C_OUT"], splits, n_use)
        kronos_real = bool(c_summ.get("kronos_real"))
        kronos_warn = c_summ.get("warning")
    for s in splits:
        data[s]["Xk"] = np.asarray(sig[s], dtype=np.float32)

    # 三合一训练：门控 MLP + Kronos 信号
    mlp, hist = B.train_model(data, d_extra=2, use_gate=True,
                              epochs=args.epochs, smoke=args.smoke, tag="d-full")

    results = {}
    all_arrays = {}
    for split in splits:
        d = data[split]
        scores = B.predict_all(mlp, d)
        met, arr = C.evaluate(scores, d["bars"], tag="d-full-%s" % split)
        results[split] = met
        all_arrays["%s_scores" % split] = scores
        for k, v in arr.items():
            all_arrays["%s_%s" % (split, k)] = v

    os.makedirs(args.out, exist_ok=True)
    mlp.save(os.path.join(args.out, "mlp_weights.npz"))
    B.MixMLP.load(os.path.join(args.out, "mlp_weights.npz"))  # 加载校验
    summary = {
        "config_name": "D-完全体 (B门控MLP + A市场门控 + C Kronos信号)",
        "B_CONFIG": {k: v for k, v in B.CONFIG.items()},
        "kronos_real": kronos_real,
        "warning": kronos_warn,
        "data": {s: {"n_use": data[s]["n_use"]} for s in splits},
        "align_check": data["_align_check"],
        "train_history": hist,
        "results": results,
        "smoke": bool(args.smoke),
        "recompute": bool(args.recompute),
        "elapsed_s": round(time.time() - t0, 1),
        "note": "冒烟模式为小样本流程验证；全量训练按 T1->T6 顺序另行安排",
    }
    C.save_summary(args.out, summary, arrays=all_arrays)
    print("[d] DONE %.0fs -> %s" % (time.time() - t0, args.out), flush=True)


if __name__ == "__main__":
    main()
