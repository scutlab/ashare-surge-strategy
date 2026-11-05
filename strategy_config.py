# 策略变量中央配置（2026-11-02）
# 所有策略、前后端的"几个股票"都从这里取，改一处即全局生效。

# --- 选股数量 ---
N_LIST = 10       # 邮件名单列出数量（每个策略）
N_STRATEGY = 3    # 策略实际交易数量；"平均每笔"按前N_STRATEGY只中成交的统计

# --- 原策略（飙升监测；2026-11-02 16:00起统一为x=0.05、+7%止盈） ---
ORIG_X_DISC = 0.05   # 买入折扣 5%（按信号收盘×0.95限价）
ORIG_STOP = 0.02     # 止损 2%
ORIG_TAKE = 0.07     # 止盈 7%（三策略一致）

# --- m策略（EV排序；2026-11-02 16:00起统一为x=0.05、+7%止盈，三策略一致，最优） ---
M_X_DISC = 0.05      # 买入折扣 5%（按信号收盘×0.95限价）
M_TAKE = 0.07        # 止盈 7%（网格搜索最优）
M_STOP = None        # 无止损
M_VALUES = [2, 3]    # 持有天数选项

# --- 模型（HistGradientBoosting，训练与回测统一） ---
HGB_CLS = dict(max_iter=300, learning_rate=0.05, max_depth=3,
               min_samples_leaf=100, class_weight="balanced",
               early_stopping=False, max_bins=128)
HGB_REG = dict(max_iter=300, learning_rate=0.05, max_depth=3,
               min_samples_leaf=100, early_stopping=False, max_bins=128)

# --- 特征（29个，与训练一致） ---
FEATS = ["vol_5","vol_20","atr_14","big_up_freq_60","big_up_freq_250","ret_1","ret_3",
         "ret_5","ret_10","ret_20","ma_bias_5","ma_bias_20","ma_bias_60",
         "close_vs_high60","close_vs_low60","breakout_20","consec_up","excess_ret_1",
         "excess_ret_5","vol_ratio_5","vol_ratio_20","is_limit_up","limit_cnt_5",
         "limit_cnt_20","mkt_ret","mkt_ret5","rsi_14","macd_diff","macd_hist"]

# --- 管道 ---
MIN_BARS_TRADING_DAY = 1000  # 当日有效行情 <1000 视为非交易日
PE_RETRY_ROUNDS = 4          # 市盈率抓取重试轮数

# --- 费用 ---
FEE = 0.005  # 每笔交易手续费 0.5%（往返），从实现收益中扣除

# --- 回测基准（Top3口径，gap-aware，已扣0.5%费） ---
# 2026-11-02 16:00起：三策略统一x=0.05、+7%止盈（网格搜索2%~10%最优）
BENCH_M2 = "+1.64%/笔"  # 70%下界+1.14%，2270笔
BENCH_M3 = "+1.80%/笔"  # 70%下界+1.33%，2371笔
