#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态选股系统 - 根据每天实时数据筛选
基于苏氏量化策略的真实计算逻辑
集成更精密的算法 (XGBoost) 进行精准评分
新增：
1. 使用 Optuna 进行 XGBoost 超参数优化，提升评分精度和区分度。
2. 引入关联规则挖掘，分析哪些条件组合更容易产生高收益，提供策略洞察。
3. 优化数据处理和输出展示。
4. 优化代码准确性、质量和效率。
5. 使用复合质量评分作为神经网络的目标变量，提高模型准确性。
6. 引入更多技术和财务指标作为模型特征。
7. 根据模型评分和规则生成明确的“短期买入”、“长期买入”和“规避”策略信号。
8. 提供模型特征重要性评估。
9. 策略推荐给出多特征组合，更具指导性。
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 导入机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb # 使用XGBoost

# 导入 Optuna 进行超参数优化
import optuna

# 导入 mlxtend 进行关联规则挖掘
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 清除代理设置
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['ALL_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

def safe_float(value_str):
    """安全地将字符串转换为浮点数，处理 '--' 和其他非数字情况"""
    try:
        if isinstance(value_str, (int, float)):
            return float(value_str)
        s = str(value_str).strip()
        if s == '--' or not s:
            return np.nan
        # 处理带单位的字符串
        if '亿' in s:
            return float(s.replace('亿', '')) * 100000000 # 1亿 = 10^8
        if '万亿' in s:
            return float(s.replace('万亿', '')) * 1000000000000 # 1万亿 = 10^12
        if '万' in s:
            return float(s.replace('万', '')) * 10000 # 1万 = 10^4
        if '%' in s: # 处理百分比
            return float(s.replace('%', ''))
        return float(s)
    except ValueError:
        return np.nan

def get_model_features(row):
    """
    根据苏氏量化策略和更多指标计算特征值，用于神经网络训练。
    返回一个包含数值特征的列表。
    """
    features = []

    # 核心苏氏策略特征 (0或1)
    # F列：价格位置条件
    low = safe_float(row.get('最低'))
    ma60 = safe_float(row.get('60日均价'))
    ma20 = safe_float(row.get('20日均价'))
    current = safe_float(row.get('最新'))

    f_condition = 0
    if pd.notna(low) and pd.notna(ma60) and ma60 > 0 and 0.85 <= low / ma60 <= 1.15:
        f_condition = 1
    elif pd.notna(current) and pd.notna(ma20) and ma20 > 0 and 0.90 <= current / ma20 <= 1.10:
        f_condition = 1
    features.append(f_condition) # Feature 0: F_价格位置

    # G列：涨幅和价格位置
    change = safe_float(row.get('涨幅%'))
    high = safe_float(row.get('最高'))
    # low = safe_float(row.get('最低')) # low已在上面获取

    g_condition = 0
    if pd.notna(change) and change >= 5.0 and pd.notna(current) and pd.notna(high) and pd.notna(low):
        if (high - low) > 0:
            threshold = high - (high - low) * 0.30
            if current >= threshold:
                g_condition = 1
        elif current == high: # 如果最高最低相同，且涨幅>=5，也算满足
            g_condition = 1
    features.append(g_condition) # Feature 1: G_涨幅位置

    # 更多技术指标特征 (数值)
    # 价格相对均线位置
    features.append(current / ma20 if pd.notna(current) and pd.notna(ma20) and ma20 > 0 else 1.0) # Feature 2: 价格_vs_MA20
    features.append(current / ma60 if pd.notna(current) and pd.notna(ma60) and ma60 > 0 else 1.0) # Feature 3: 价格_vs_MA60
    features.append(ma20 / ma60 if pd.notna(ma20) and pd.notna(ma60) and ma60 > 0 else 1.0) # Feature 4: MA20_vs_MA60

    # 每日波动幅度
    daily_range_ratio = (high - low) / current if pd.notna(high) and pd.notna(low) and pd.notna(current) and current > 0 else 0.0
    features.append(daily_range_ratio) # Feature 5: 日内波动率

    # 收盘价在日内区间的位置 (越接近最高价越强)
    close_pos_in_range = (current - low) / (high - low) if pd.notna(current) and pd.notna(low) and pd.notna(high) and (high - low) > 0 else 0.5
    features.append(close_pos_in_range) # Feature 6: 收盘价_日内位置

    # 财务与市场指标特征 (数值)
    profit = safe_float(row.get('归属净利润')) # 单位是亿，这里保持原值
    features.append(profit if pd.notna(profit) else 0) # Feature 7: 归属净利润

    turnover = safe_float(row.get('实际换手%'))
    features.append(turnover if pd.notna(turnover) else 0) # Feature 8: 实际换手率

    market_cap = safe_float(row.get('总市值')) # 单位是亿，这里保持原值
    features.append(market_cap if pd.notna(market_cap) else 0) # Feature 9: 总市值

    pe_ratio = safe_float(row.get('市盈率(动)'))
    features.append(pe_ratio if pd.notna(pe_ratio) else 1000) # Feature 10: 市盈率(动) (缺失时给一个大值，表示高估)

    # 成交额 (单位是亿)
    turnover_value = safe_float(row.get('成交额')) # akshare返回的是亿，这里保持原值
    features.append(turnover_value if pd.notna(turnover_value) else 0) # Feature 11: 成交额

    # 涨幅%
    features.append(change if pd.notna(change) else 0) # Feature 12: 涨幅%

    return features

def objective(trial, X_train, y_train, X_test, y_test):
    """
    Optuna 优化目标函数 for XGBoost
    """
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'seed': 42,
        'n_jobs': -1,
    }

    if param['booster'] == 'gbtree' or param['booster'] == 'dart':
        param['max_depth'] = trial.suggest_int('max_depth', 3, 9)
        param['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
        param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
        param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

    if param['booster'] == 'dart':
        param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        param['rate_drop'] = trial.suggest_uniform('rate_drop', 0.0, 1.0)
        param['skip_drop'] = trial.suggest_uniform('skip_drop', 0.0, 1.0)

    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              early_stopping_rounds=50, # 增加耐心
              verbose=False)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse # Optuna 默认最小化目标

def train_xgboost_model(df):
    """
    训练 XGBoost 模型，预测股票评分，使用 Optuna 进行超参数优化。
    使用复合质量评分作为目标变量。
    """
    print("\n   准备训练数据...")
    X = []
    for _, row in df.iterrows():
        features = get_model_features(row)
        X.append(features)

    X = np.array(X)

    # 提取用于计算质量评分的列 (原始数值，未格式化)
    change = df['涨幅%'].apply(safe_float)
    profit = df['归属净利润'].apply(safe_float)
    turnover = df['实际换手%'].apply(safe_float)
    market_cap = df['总市值'].apply(safe_float)
    pe_ratio = df['市盈率(动)'].apply(safe_float)
    current_price = df['最新'].apply(safe_float)
    ma20 = df['20日均价'].apply(safe_float)

    # 归一化各个指标 (使用 min-max 归一化，处理NaN)
    # 对于涨幅、净利润、市值，越大越好
    change_norm = (change - change.min()) / (change.max() - change.min()) if change.max() != change.min() else 0
    profit_norm = (profit - profit.min()) / (profit.max() - profit.min()) if profit.max() != profit.min() else 0
    market_cap_norm = (market_cap - market_cap.min()) / (market_cap.max() - market_cap.min()) if market_cap.max() != market_cap.min() else 0

    # 换手率：适中最好，偏离0.5越远越差
    turnover_norm = (turnover - turnover.min()) / (turnover.max() - turnover.min()) if turnover.max() != turnover.min() else 0
    turnover_score = 1 - abs(turnover_norm - 0.5) * 2 # 0.5时为1，0或1时为0

    # 市盈率：越低越好，但要避免负值和过高值
    pe_ratio_capped = pe_ratio.apply(lambda x: min(x, 100) if x > 0 else 100) # 将过高或负的市盈率限制在合理范围
    pe_ratio_norm = (pe_ratio_capped - pe_ratio_capped.min()) / (pe_ratio_capped.max() - pe_ratio_capped.min()) if pe_ratio_capped.max() != pe_ratio_capped.min() else 0
    pe_score = 1 - pe_ratio_norm

    # 价格相对20日均线位置：越接近均线越好，但略高于均线更佳
    price_ma20_ratio = current_price / ma20
    # 之前有错，现在修正
    price_ma20_ratio_score = price_ma20_ratio.apply(lambda x: max(0, 1 - abs(x - 1))) # 越接近1分越高，但不能为负

    # 处理 NaN 值，用 0 填充
    change_norm = change_norm.fillna(0)
    profit_norm = profit_norm.fillna(0)
    turnover_score = turnover_score.fillna(0)
    market_cap_norm = market_cap_norm.fillna(0)
    pe_score = pe_score.fillna(0)
    price_ma20_ratio_score = price_ma20_ratio_score.fillna(0)

    # 计算复合质量评分 (可以调整权重)
    df['quality_score'] = (
        0.25 * change_norm +
        0.20 * profit_norm +
        0.15 * turnover_score +
        0.15 * market_cap_norm +
        0.15 * pe_score +
        0.10 * price_ma20_ratio_score
    )

    y = df['quality_score'].values

    # 移除包含 NaN 或无穷大的行
    mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1) & ~np.isnan(y) & ~np.isinf(y)
    X = X[mask]
    y = y[mask]

    if len(X) < 20: # 至少需要一些数据来划分训练集和测试集
        print("   ❌ 有效训练数据不足，无法训练模型。")
        return None, None, None # 返回模型、Scaler、特征名称

    print(f"   有效训练样本数: {len(X)}")

    # 数据预处理
    print("   数据预处理 (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    print("   划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Optuna 超参数优化
    print("   启动 Optuna 超参数优化 (可能需要一些时间)...")
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    try:
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50, show_progress_bar=True)
    except Exception as e:
        print(f"   Optuna 优化过程中发生错误: {e}")
        print("   将使用默认或预设参数训练模型。")
        # 如果Optuna失败，使用一个合理的默认配置
        best_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'lambda': 1.0,
            'alpha': 0.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'max_depth': 6,
            'eta': 0.1,
            'gamma': 0.0,
            'grow_policy': 'depthwise',
            'seed': 42,
            'n_jobs': -1,
        }
    else:
        print("\n   Optuna 优化完成。")
        print(f"   最佳均方误差 (MSE): {study.best_value:.4f}")
        print(f"   最佳超参数: {study.best_params}")
        best_params = study.best_params

    # 使用最佳参数训练最终模型
    print("   使用最佳参数训练最终 XGBoost 模型...")
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              early_stopping_rounds=100, # 增加耐心
              verbose=False)

    # 评估模型
    print("   评估最终模型...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"   最终模型均方误差 (MSE): {mse:.4f}")
    print(f"   最终模型R²分数: {r2:.4f}")

    # 获取特征名称 (与get_model_features中的顺序一致)
    feature_names = [
        "F_价格位置", "G_涨幅位置", "价格_vs_MA20", "价格_vs_MA60", "MA20_vs_MA60",
        "日内波动率", "收盘价_日内位置", "归属净利润", "实际换手率", "总市值", "市盈率(动)",
        "成交额", "涨幅%"
    ]

    return model, scaler, feature_names

def predict_score_with_model(row, model, scaler):
    """
    使用训练好的模型预测股票评分
    """
    features = get_model_features(row)
    if any(pd.isna(f) or np.isinf(f) for f in features):
        return np.nan

    features = np.array(features).reshape(1, -1)
    try:
        features_scaled = scaler.transform(features)
        score = model.predict(features_scaled)[0]
        return score
    except Exception as e:
        # print(f"预测分数时发生错误: {e}, 特征: {features}")
        return np.nan

def generate_strategy_signals(stock_data, nn_score):
    """
    根据股票数据和神经网络评分生成策略信号。
    """
    signals = []
    current = safe_float(stock_data.get('最新'))
    change = safe_float(stock_data.get('涨幅%'))
    turnover = safe_float(stock_data.get('实际换手%'))
    ma20 = safe_float(stock_data.get('20日均价'))
    ma60 = safe_float(stock_data.get('60日均价'))
    pe_ratio = safe_float(stock_data.get('市盈率(动)'))
    profit = safe_float(stock_data.get('归属净利润'))
    market_cap = safe_float(stock_data.get('总市值'))
    high = safe_float(stock_data.get('最高'))
    low = safe_float(stock_data.get('最低'))
    成交额 = safe_float(stock_data.get('成交额'))

    # 短期买入信号条件列表
    short_term_buy_conditions = []
    if pd.notna(nn_score) and nn_score > 0.7:
        short_term_buy_conditions.append("NN高分(>0.7)")
    if pd.notna(change) and change >= 2.0:
        short_term_buy_conditions.append(f"涨幅积极(>={change:.2f}%)")
    if pd.notna(current) and pd.notna(ma20) and ma20 > 0 and current > ma20 * 1.01:
        short_term_buy_conditions.append("站上20MA(>1.01倍)")
    if pd.notna(turnover) and 1.0 < turnover < 15.0:
        short_term_buy_conditions.append(f"换手适中({turnover:.2f}%)")
    if pd.notna(current) and pd.notna(high) and pd.notna(low) and (high - low) > 0 and (current - low) / (high - low) > 0.7:
        short_term_buy_conditions.append("收盘强势(近高点)")
    if pd.notna(成交额) and 成交额 > 5.0: # 成交额大于5亿
        short_term_buy_conditions.append(f"成交活跃(>{成交额:.2f}亿)")

    # 判断短期买入信号
    if len(short_term_buy_conditions) >= 3: # 满足至少3个条件
        signals.append(f"短期买入 ({', '.join(short_term_buy_conditions)})")

    # 长期买入信号条件列表
    long_term_buy_conditions = []
    if pd.notna(nn_score) and nn_score > 0.6:
        long_term_buy_conditions.append("NN中高分(>0.6)")
    if pd.notna(pe_ratio) and 0 < pe_ratio < 40:
        long_term_buy_conditions.append(f"PE合理({pe_ratio:.2f})")
    if pd.notna(profit) and profit > 0.5: # 归属净利润大于5000万 (0.5亿)
        long_term_buy_conditions.append(f"净利润良好(>{profit:.2f}亿)")
    if pd.notna(market_cap) and market_cap > 100: # 总市值大于100亿
        long_term_buy_conditions.append(f"市值较大(>{market_cap:.2f}亿)")
    if pd.notna(current) and pd.notna(ma60) and ma60 > 0 and current > ma60 * 0.95:
        long_term_buy_conditions.append("价格近60MA(>0.95倍)")
    if pd.notna(ma20) and pd.notna(ma60) and ma60 > 0 and ma20 > ma60: # 20日均线在60日均线之上
        long_term_buy_conditions.append("均线多头排列(20MA>60MA)")

    # 判断长期买入信号
    if len(long_term_buy_conditions) >= 3: # 满足至少3个条件
        signals.append(f"长期买入 ({', '.join(long_term_buy_conditions)})")

    # 规避/警示信号条件列表
    avoid_conditions = []
    if pd.notna(change) and change < -5.0:
        avoid_conditions.append(f"大幅下跌(<{change:.2f}%)")
    if pd.notna(turnover) and turnover > 25.0:
        avoid_conditions.append(f"换手过高(>{turnover:.2f}%)")
    if pd.notna(pe_ratio) and (pe_ratio < 0 or pe_ratio > 150):
        avoid_conditions.append(f"PE异常({pe_ratio:.2f})")
    if pd.notna(profit) and profit < 0:
        avoid_conditions.append(f"净利润为负({profit:.2f}亿)")
    if pd.notna(current) and pd.notna(ma20) and ma20 > 0 and current < ma20 * 0.95:
        avoid_conditions.append("跌破20MA(<0.95倍)")
    if pd.notna(market_cap) and market_cap < 30: # 总市值小于30亿
        avoid_conditions.append(f"市值过小(<{market_cap:.2f}亿)")

    # 判断规避/警示信号
    if len(avoid_conditions) >= 2: # 满足至少2个条件
        signals.append(f"规避/警示 ({', '.join(avoid_conditions)})")

    if not signals:
        signals.append("无明确信号")

    return "; ".join(signals)

def perform_association_rule_mining(df):
    """
    使用关联规则挖掘来发现苏氏量化策略条件与高收益之间的关系。
    """
    print("\n4. 执行关联规则挖掘...")

    # 准备数据：将特征和目标变量离散化
    data_for_ar = []
    for _, row in df.iterrows():
        # 使用 get_model_features 获取原始数值特征
        raw_features = get_model_features(row)
        items = []

        # 苏氏策略特征
        if raw_features[0] == 1: items.append("F_价格位置_满足")
        else: items.append("F_价格位置_不满足")

        if raw_features[1] == 1: items.append("G_涨幅位置_满足")
        else: items.append("G_涨幅位置_不满足")

        # 归属净利润 (Feature 7): 分为高、中、低三档
        profit = raw_features[7]
        if profit >= 10: items.append("H_净利润_高(>=10亿)")
        elif 1 <= profit < 10: items.append("H_净利润_中(1-10亿)")
        else: items.append("H_净利润_低(<1亿)")

        # 实际换手率 (Feature 8): 分为高、低两档
        turnover = raw_features[8]
        if turnover >= 10: items.append("I_换手率_高(>=10%)")
        else: items.append("I_换手率_低(<10%)")

        # 总市值 (Feature 9): 分为大、中、小三档
        market_cap = raw_features[9]
        if market_cap >= 500: items.append("J_市值_大(>=500亿)")
        elif 100 <= market_cap < 500: items.append("J_市值_中(100-500亿)")
        else: items.append("J_市值_小(<100亿)")

        # 价格相对20日均线位置 (Feature 2): 分为高于、接近、低于三档
        price_vs_ma20 = raw_features[2]
        if price_vs_ma20 > 1.05: items.append("价格_远高于20MA")
        elif 0.95 <= price_vs_ma20 <= 1.05: items.append("价格_近20MA")
        else: items.append("价格_远低于20MA")

        # 收盘价在日内区间的位置 (Feature 6): 分为强势、弱势两档
        close_pos_in_range = raw_features[6]
        if close_pos_in_range > 0.7: items.append("收盘价_日内强势")
        else: items.append("收盘价_日内弱势")

        # 目标变量：高涨幅 (例如，涨幅 > 3%)
        change = safe_float(row.get('涨幅%'))
        if pd.notna(change) and change > 3.0: # 可以调整这个阈值
            items.append("高涨幅")
        else:
            items.append("低涨幅")

        data_for_ar.append(items)

    if not data_for_ar:
        print("   ❌ 没有足够的数据进行关联规则挖掘。")
        return

    te = TransactionEncoder()
    te_ary = te.fit(data_for_ar).transform(data_for_ar)
    df_ar = pd.DataFrame(te_ary, columns=te.columns_)

    # 查找频繁项集
    # min_support 可以根据数据量调整，太小规则太多，太大规则太少
    frequent_itemsets = apriori(df_ar, min_support=0.01, use_colnames=True) # 降低支持度以发现更多规则
    if frequent_itemsets.empty:
        print("   ⚠️ 未找到频繁项集，请尝试降低 min_support。")
        return

    # 生成关联规则
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1) # 提高提升度阈值
    if rules.empty:
        print("   ⚠️ 未找到有意义的关联规则，请尝试降低 min_threshold 或检查数据。")
        return

    # 筛选并打印与“高涨幅”相关的规则
    high_return_rules = rules[rules['consequents'].apply(lambda x: '高涨幅' in x)]
    high_return_rules = high_return_rules.sort_values(by=['lift', 'confidence'], ascending=False)

    print("\n   发现以下与 '高涨幅' 相关的关联规则 (按 Lift 降序):")
    if high_return_rules.empty:
        print("   未找到直接导致 '高涨幅' 的关联规则。")
    else:
        for i, rule in high_return_rules.head(10).iterrows(): # 只显示前10条
            antecedent_str = ', '.join(list(rule['antecedents']))
            consequent_str = ', '.join(list(rule['consequents']))
            print(f"   规则 {i+1}: {antecedent_str} => {consequent_str}")
            print(f"     支持度 (Support): {rule['support']:.4f}")
            print(f"     置信度 (Confidence): {rule['confidence']:.4f}")
            print(f"     提升度 (Lift): {rule['lift']:.4f}")
            print("-" * 40)

    print("\n   关联规则挖掘完成。这些规则可以为策略优化提供洞察。")


def main():
    """主程序"""
    print("\n" + "="*60)
    print("动态选股系统 - 实时计算版 (集成XGBoost与关联规则)")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 创建输出目录
    os.makedirs('输出数据', exist_ok=True)
    os.makedirs('参考数据', exist_ok=True) # 确保参考数据目录存在

    # ========== 第一步：获取数据 ==========
    print("\n1. 获取A股数据...")

    df = pd.DataFrame()
    # 尝试获取实时数据
    try:
        print("   尝试获取实时数据 (akshare.stock_zh_a_spot_em)...")
        df_realtime = ak.stock_zh_a_spot_em()
        print(f"   ✅ 成功获取 {len(df_realtime)} 只股票的实时数据")

        # 统一列名
        df_realtime.rename(columns={
            '最新价': '最新',
            '涨跌幅': '涨幅%',
            '换手率': '实际换手%',
            '市盈率-动态': '市盈率(动)',
            '成交额': '成交额' # 确保成交额列名正确
        }, inplace=True)

        # 保存原始代码
        df_realtime['原始代码'] = df_realtime['代码'].copy()

        # 格式化代码以便Excel识别为文本
        df_realtime['代码'] = df_realtime['代码'].apply(lambda x: f'= "{str(x)}"')

        # 确保所有关键列存在，并初始化为NaN
        required_cols = ['代码', '名称', '最新', '涨幅%', '最高', '最低', '实际换手%', '成交额',
                         '所属行业', '20日均价', '60日均价', '市盈率(动)','总市值', '归属净利润', '昨收', '开盘']
        for col in required_cols:
            if col not in df_realtime.columns:
                df_realtime[col] = np.nan

        df = df_realtime.copy()

    except Exception as e:
        print(f"   ❌ 获取实时数据失败: {e}")
        df = pd.DataFrame()

    # 尝试从本地 Excel 文件补充数据
    try:
        table_path = os.path.join('参考数据', 'Table.xls')
        if os.path.exists(table_path):
            print("   尝试从本地 Table.xls 补充数据...")
            df_table = pd.read_csv(table_path, sep='\t', encoding='gbk') # 确保编码正确
            # 统一代码格式
            df_table['代码'] = df_table['代码'].astype(str).str.replace('="', '').str.replace('"', '')

            # 将df_table中的'代码'列设置为字符串类型
            df_table['代码'] = df_table['代码'].astype(str)

            # 将df中的'代码'列设置为字符串类型
            df['代码'] = df['代码'].astype(str).str.replace('="', '').str.replace('"', '')

            # 确保df_table中的列存在于df中
            for col in ['20日均价', '60日均价', '市盈率(动)', '总市值', '归属净利润']:
                if col not in df_table.columns:
                    df_table[col] = np.nan

            # 补充缺失数据
            merged_df = pd.merge(df, df_table[['代码', '20日均价', '60日均价', '市盈率(动)', '总市值', '归属净利润']],
                                  on='代码', how='left', suffixes=('', '_table'))

            for col in ['20日均价', '60日均价', '市盈率(动)', '总市值', '归属净利润']:
                df[col] = merged_df[col + '_table'].fillna(merged_df[col])

            print(f"   ✅ 补充了 {len(df)} 条参考数据中的缺失信息 (如果Table.xls存在且有匹配数据)")
        else:
            print("   ⚠️ 未找到 Table.xls，跳过数据补充")
    except Exception as e:
        print(f"   ❌ 从 Table.xls 补充数据失败: {e}")

    # 保存A股数据到CSV
    csv_path = os.path.join('输出数据', 'A股数据.csv')
    df.to_csv(csv_path, encoding='utf_8_sig', index=True, header=True)
    print(f"\n✅ A股数据已保存: {csv_path}")
    print(f"   共 {len(df)} 只股票")

    # ========== 第二步：训练 XGBoost 模型 ==========
    model, scaler, feature_names = train_xgboost_model(df.copy()) # 传递 df 的副本

    if model is None or scaler is None:
        print("   ❌ 模型训练失败，无法进行后续预测。")
        return

    # 显示模型特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        print("\n   模型特征重要性 (Feature Importance):")
        print(feature_importance.to_string(index=False))
    else:
        print("\n   ⚠️ 模型不支持特征重要性评估。")

    # ========== 第三步：动态筛选优质股票并生成策略信号 ==========
    print("\n3. 动态筛选优质股票并生成策略信号 (基于XGBoost评分)...")

    # 使用模型预测评分
    df['predicted_score'] = df.apply(lambda row: predict_score_with_model(row, model, scaler), axis=1)

    # 移除 predicted_score 为 NaN 的行
    df = df.dropna(subset=['predicted_score']).copy()

    # 根据 predicted_score 排序
    df = df.sort_values(by='predicted_score', ascending=False)

    # 确定最低优质率阈值 (基于前20名或全部)
    num_to_consider = min(20, len(df))
    min_quality_threshold = df['predicted_score'].iloc[num_to_consider - 1] if num_to_consider > 0 else 0

    # 筛选符合条件的股票
    qualified_stocks = df[df['predicted_score'] >= min_quality_threshold].copy()

    # 生成策略信号
    qualified_stocks['strategy_signal'] = qualified_stocks.apply(lambda row: generate_strategy_signals(row, row['predicted_score']), axis=1)

    # 保存优质股票及策略信号到文本文件
    output_path = os.path.join('输出数据', '优质股票_策略信号.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("苏氏量化策略 - 优质股票筛选结果与策略信号 (XGBoost评分)\n")
        f.write(f"筛选时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"最低优质率阈值 (基于前20名或全部): {min_quality_threshold:.4f}\n")

        buy_signals = qualified_stocks[qualified_stocks['strategy_signal'].str.contains('买入')].copy()
        num_buy_signals = len(buy_signals)
        f.write(f"符合买入信号的优质股票数量: {num_buy_signals}\n")

        f.write("=" * 60 + "\n")
        for index, row in buy_signals.iterrows():
            f.write(f"股票代码: {row['代码']}\n")
            f.write(f"股票名称: {row['名称']}\n")
            f.write(f"所属行业: {row['所属行业']}\n")
            f.write(f"优质率 (XGBoost评分): {row['predicted_score']:.4f}\n")
            f.write(f"今日涨幅: {row['涨幅%']}\n")
            f.write(f"策略信号: {row['strategy_signal']}\n")
            f.write("-" * 30 + "\n")

    print(f"\n✅ 优质股票及策略信号已保存: {output_path}")
    print(f"   找到 {num_buy_signals} 只符合买入条件的优质股票（最低优质率={min_quality_threshold:.4f}）")

    # 输出今日优质股票列表 (带买入信号的前 N 名)
    N = min(4, num_buy_signals)
    print(f"\n🎯 今日优质股票列表 (前{N}名，仅显示买入信号)：")
    print("=" * 85)
    print(f"{'股票代码':<10} {'股票名称':<15} {'涨幅':<8} {'优质率':<8} {'所属行业':<15} {'策略信号':<40}")
    print("-" * 85)
    for index, row in buy_signals.head(N).iterrows():
        print(f"{row['代码']:<10} {row['名称']:<15} {str(row['涨幅%']):<8} {row['predicted_score']:.4f:<8} {row['所属行业']:<15} {row['strategy_signal']:<40}")
    print("=" * 85)

    # ========== 第四步：执行关联规则挖掘 ==========
    perform_association_rule_mining(df.copy()) # 传递 df 的副本

    print("\n" + "="*60)
    print("✅ 程序执行完成！")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

