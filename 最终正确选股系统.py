#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态选股系统 - 根据每天实时数据筛选
基于苏氏量化策略的真实计算逻辑
集成神经网络进行精准评分
新增：
1. 使用 Optuna 进行神经网络超参数优化，提升评分精度和区分度。
2. 引入关联规则挖掘，分析哪些条件组合更容易产生高收益，提供策略洞察。
3. 优化数据处理和输出展示。
4. 优化代码准确性、质量和效率。
5. 使用复合质量评分作为神经网络的目标变量，提高模型准确性。
6. **重大升级：引入更多技术指标，并定义短期/长期买卖策略。**
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# 导入神经网络相关库
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler # 新增MinMaxScaler用于目标变量归一化
from sklearn.metrics import mean_squared_error, r2_score

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

# 全局变量，用于存储历史数据，避免重复获取
GLOBAL_HIST_DATA = {}

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
            return float(s.replace('亿', '')) * 10000
        if '万亿' in s:
            return float(s.replace('万亿', '')) * 100000000
        if '万' in s:
            return float(s.replace('万', '')) / 10000
        return float(s)
    except ValueError:
        return np.nan

def get_stock_history_data(symbol, start_date, end_date):
    """
    获取单只股票的历史行情数据，并缓存。
    symbol: 股票代码，如 '000001'
    start_date, end_date: 日期字符串 'YYYYMMDD'
    """
    if symbol in GLOBAL_HIST_DATA:
        # 检查缓存数据是否覆盖所需日期范围
        cached_df = GLOBAL_HIST_DATA[symbol]
        if not cached_df.empty and \
           pd.to_datetime(cached_df['日期'].min()) <= pd.to_datetime(start_date) and \
           pd.to_datetime(cached_df['日期'].max()) >= pd.to_datetime(end_date):
            return cached_df[(cached_df['日期'] >= start_date) & (cached_df['日期'] <= end_date)].copy()

    try:
        # 尝试从 akshare 获取数据
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if df.empty:
            print(f"   ⚠️ 未获取到 {symbol} 的历史数据。")
            return pd.DataFrame()
        df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        df['日期'] = df['日期'].dt.strftime('%Y%m%d')
        df = df.sort_values(by='日期').reset_index(drop=True)
        GLOBAL_HIST_DATA[symbol] = df.copy() # 缓存数据
        return df
    except Exception as e:
        print(f"   ❌ 获取 {symbol} 历史数据失败: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df_hist):
    """
    计算技术指标，并返回最新一天的指标值。
    df_hist: 包含历史行情数据的DataFrame，至少包含 '收盘', '最高', '最低', '成交量'
    """
    if df_hist.empty or len(df_hist) < 200: # 至少需要200天数据来计算长期均线和布林带
        return {
            'MA5': np.nan, 'MA10': np.nan, 'MA20': np.nan, 'MA60': np.nan, 'MA120': np.nan, 'MA200': np.nan,
            'RSI': np.nan, 'MACD_DIF': np.nan, 'MACD_DEA': np.nan, 'MACD_HIST': np.nan,
            'BOLL_UP': np.nan, 'BOLL_MID': np.nan, 'BOLL_LOW': np.nan,
            'VOL_MA5': np.nan, 'VOL_MA10': np.nan, 'VOL_CHANGE': np.nan
        }

    # 确保数据类型正确
    df_hist['收盘'] = pd.to_numeric(df_hist['收盘'], errors='coerce')
    df_hist['最高'] = pd.to_numeric(df_hist['最高'], errors='coerce')
    df_hist['最低'] = pd.to_numeric(df_hist['最低'], errors='coerce')
    df_hist['成交量'] = pd.to_numeric(df_hist['成交量'], errors='coerce')
    df_hist['涨跌幅'] = pd.to_numeric(df_hist['涨跌幅'], errors='coerce')

    # 移动平均线 (MA)
    df_hist['MA5'] = df_hist['收盘'].rolling(window=5).mean()
    df_hist['MA10'] = df_hist['收盘'].rolling(window=10).mean()
    df_hist['MA20'] = df_hist['收盘'].rolling(window=20).mean()
    df_hist['MA60'] = df_hist['收盘'].rolling(window=60).mean()
    df_hist['MA120'] = df_hist['收盘'].rolling(window=120).mean()
    df_hist['MA200'] = df_hist['收盘'].rolling(window=200).mean()

    # 相对强弱指数 (RSI)
    delta = df_hist['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_hist['RSI'] = 100 - (100 / (1 + rs))

    # 异同移动平均线 (MACD)
    exp1 = df_hist['收盘'].ewm(span=12, adjust=False).mean()
    exp2 = df_hist['收盘'].ewm(span=26, adjust=False).mean()
    df_hist['MACD_DIF'] = exp1 - exp2
    df_hist['MACD_DEA'] = df_hist['MACD_DIF'].ewm(span=9, adjust=False).mean()
    df_hist['MACD_HIST'] = (df_hist['MACD_DIF'] - df_hist['MACD_DEA']) * 2 # MACD柱

    # 布林带 (Bollinger Bands)
    df_hist['BOLL_MID'] = df_hist['收盘'].rolling(window=20).mean()
    df_hist['BOLL_STD'] = df_hist['收盘'].rolling(window=20).std()
    df_hist['BOLL_UP'] = df_hist['BOLL_MID'] + (df_hist['BOLL_STD'] * 2)
    df_hist['BOLL_LOW'] = df_hist['BOLL_MID'] - (df_hist['BOLL_STD'] * 2)

    # 成交量均线
    df_hist['VOL_MA5'] = df_hist['成交量'].rolling(window=5).mean()
    df_hist['VOL_MA10'] = df_hist['成交量'].rolling(window=10).mean()
    df_hist['VOL_CHANGE'] = df_hist['成交量'].pct_change() # 成交量变化率

    # 获取最新一天的指标值
    latest_data = df_hist.iloc[-1]
    
    return {
        'MA5': latest_data.get('MA5'), 'MA10': latest_data.get('MA10'), 'MA20': latest_data.get('MA20'),
        'MA60': latest_data.get('MA60'), 'MA120': latest_data.get('MA120'), 'MA200': latest_data.get('MA200'),
        'RSI': latest_data.get('RSI'),
        'MACD_DIF': latest_data.get('MACD_DIF'), 'MACD_DEA': latest_data.get('MACD_DEA'), 'MACD_HIST': latest_data.get('MACD_HIST'),
        'BOLL_UP': latest_data.get('BOLL_UP'), 'BOLL_MID': latest_data.get('BOLL_MID'), 'BOLL_LOW': latest_data.get('BOLL_LOW'),
        'VOL_MA5': latest_data.get('VOL_MA5'), 'VOL_MA10': latest_data.get('VOL_MA10'), 'VOL_CHANGE': latest_data.get('VOL_CHANGE')
    }

def calculate_all_features(row, hist_data_map):
    """
    根据苏氏量化策略、基本面和技术指标计算特征值。
    返回一个包含数值特征的列表。
    """
    features = []

    # 1. 苏氏量化策略特征 (F, G)
    # F列：价格位置条件 (0或1)
    try:
        low = safe_float(row.get('最低'))
        ma60 = safe_float(row.get('60日均价'))
        ma20 = safe_float(row.get('20日均价'))
        current = safe_float(row.get('最新'))

        f_condition = 0
        if pd.notna(low) and pd.notna(ma60) and ma60 > 0 and 0.85 <= low / ma60 <= 1.15:
            f_condition = 1
        elif pd.notna(current) and pd.notna(ma20) and ma20 > 0 and 0.90 <= current / ma20 <= 1.10:
            f_condition = 1
        features.append(f_condition)
    except:
        features.append(0) # 默认值

    # G列：涨幅和价格位置 (0或1)
    try:
        change = safe_float(row.get('涨幅%'))
        current = safe_float(row.get('最新'))
        high = safe_float(row.get('最高'))
        low = safe_float(row.get('最低'))

        g_condition = 0
        if pd.notna(change) and change >= 5.0 and pd.notna(current) and pd.notna(high) and pd.notna(low):
            if (high - low) > 0: # 避免除以零
                threshold = high - (high - low) * 0.30
                if current >= threshold:
                    g_condition = 1
            elif current == high: # 如果最高最低相同，且涨幅>=5，也算满足
                g_condition = 1
        features.append(g_condition)
    except:
        features.append(0) # 默认值

    # 2. 基本面特征 (H, I, J)
    # H列：归属净利润 (数值，单位亿)
    try:
        profit = safe_float(row.get('归属净利润'))
        features.append(profit if pd.notna(profit) else 0)
    except:
        features.append(0)

    # I列：实际换手率 (数值)
    try:
        turnover = safe_float(row.get('实际换手%'))
        features.append(turnover if pd.notna(turnover) else 0) # 缺失时给0
    except:
        features.append(0)

    # J列：总市值 (数值，单位亿)
    try:
        cap = safe_float(row.get('总市值'))
        features.append(cap if pd.notna(cap) else 0)
    except:
        features.append(0)

    # K列：市盈率(动) (数值)
    try:
        pe = safe_float(row.get('市盈率(动)'))
        features.append(pe if pd.notna(pe) else 0) # 缺失时给0
    except:
        features.append(0)

    # 3. 技术指标特征 (从 hist_data_map 中获取)
    symbol = row.get('原始代码')
    tech_indicators = hist_data_map.get(symbol, {})

    features.append(tech_indicators.get('MA5', np.nan))
    features.append(tech_indicators.get('MA10', np.nan))
    features.append(tech_indicators.get('MA20', np.nan))
    features.append(tech_indicators.get('MA60', np.nan))
    features.append(tech_indicators.get('MA120', np.nan))
    features.append(tech_indicators.get('MA200', np.nan))
    features.append(tech_indicators.get('RSI', np.nan))
    features.append(tech_indicators.get('MACD_DIF', np.nan))
    features.append(tech_indicators.get('MACD_DEA', np.nan))
    features.append(tech_indicators.get('MACD_HIST', np.nan))
    features.append(tech_indicators.get('BOLL_UP', np.nan))
    features.append(tech_indicators.get('BOLL_MID', np.nan))
    features.append(tech_indicators.get('BOLL_LOW', np.nan))
    features.append(tech_indicators.get('VOL_MA5', np.nan))
    features.append(tech_indicators.get('VOL_MA10', np.nan))
    features.append(tech_indicators.get('VOL_CHANGE', np.nan))

    # 确保所有特征都是数值类型，并处理NaN
    final_features = [f if pd.notna(f) else 0 for f in features] # 将所有NaN填充为0

    return final_features

def objective(trial, X_train, y_train, X_test, y_test):
    """
    Optuna 优化目标函数
    """
    hidden_layer_sizes = []
    n_layers = trial.suggest_int('n_layers', 1, 3)
    for i in range(n_layers):
        hidden_layer_sizes.append(trial.suggest_int(f'n_units_l{i}', 32, 256)) # 增加神经元数量范围

    activation = trial.suggest_categorical('activation', ['relu', 'tanh']) # 移除logistic，relu和tanh通常表现更好
    solver = trial.suggest_categorical('solver', ['adam']) # 简化为adam，通常效果最好
    alpha = trial.suggest_loguniform('alpha', 1e-6, 1e-2) # 调整正则化强度范围
    learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-2)

    model = MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        random_state=42,
        max_iter=1000, # 增加最大迭代次数
        early_stopping=True,
        n_iter_no_change=30, # 增加耐心
        tol=1e-5 # 增加容忍度
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse # Optuna 默认最小化目标

def train_neural_network(df):
    """
    训练神经网络模型，预测股票评分，使用 Optuna 进行超参数优化。
    使用复合质量评分作为目标变量。
    """
    print("\n   准备训练数据...")
    X = []
    
    # 获取所有股票的历史数据，用于计算技术指标
    today_str = datetime.now().strftime('%Y%m%d')
    # 需要足够长的历史数据来计算所有指标，例如200天均线
    start_date_hist = (datetime.now() - timedelta(days=300)).strftime('%Y%m%d')
    
    hist_data_map = {}
    for symbol in df['原始代码'].unique():
        hist_data_map[symbol] = calculate_technical_indicators(
            get_stock_history_data(symbol, start_date_hist, today_str)
        )

    for _, row in df.iterrows():
        features = calculate_all_features(row, hist_data_map)
        X.append(features)

    X = np.array(X)

    # 提取用于计算复合质量评分的列 (使用原始数值，未格式化的df)
    df_raw_values = df.copy()
    for col in ['涨幅%', '归属净利润', '实际换手%', '总市值', '市盈率(动)']:
        df_raw_values[col] = df_raw_values[col].apply(safe_float)

    change = df_raw_values['涨幅%']
    profit = df_raw_values['归属净利润']
    turnover = df_raw_values['实际换手%']
    market_cap = df_raw_values['总市值']
    pe_ratio = df_raw_values['市盈率(动)']

    # 归一化各个指标 (使用 MinMaxScaler，处理NaN值)
    # 涨幅：越高越好
    change_norm = MinMaxScaler().fit_transform(change.fillna(change.median()).values.reshape(-1, 1)).flatten()
    # 净利润：越高越好
    profit_norm = MinMaxScaler().fit_transform(profit.fillna(profit.median()).values.reshape(-1, 1)).flatten()
    # 换手率：适中最好，过高或过低都不好。这里简单处理为越低越好，或者可以设计一个二次函数
    # 暂时保持越低越好，但可以根据策略调整
    turnover_norm = MinMaxScaler().fit_transform(turnover.fillna(turnover.median()).values.reshape(-1, 1)).flatten()
    # 市值：越大越好 (倾向于大中盘股)
    market_cap_norm = MinMaxScaler().fit_transform(market_cap.fillna(market_cap.median()).values.reshape(-1, 1)).flatten()
    # 市盈率：越低越好 (但要避免负值或过高异常值)
    # 对PE进行特殊处理，避免负值和极端高值影响归一化
    pe_ratio_filtered = pe_ratio.copy()
    pe_ratio_filtered[pe_ratio_filtered <= 0] = np.nan # 负PE通常表示亏损，不参与正常估值
    pe_ratio_filtered[pe_ratio_filtered > 500] = 500 # 限制极端高值
    pe_ratio_norm = MinMaxScaler().fit_transform(pe_ratio_filtered.fillna(pe_ratio_filtered.median()).values.reshape(-1, 1)).flatten()
    pe_ratio_norm = 1 - pe_ratio_norm # 越低越好，所以1-归一化值

    # 计算复合质量评分 (可以调整权重，这里更侧重基本面和合理估值)
    # 权重分配：
    # 涨幅 (短期动量): 0.15
    # 净利润 (盈利能力): 0.30
    # 换手率 (流动性/活跃度，这里假设适中偏低为好): 0.10
    # 市值 (规模/稳定性): 0.25
    # 市盈率 (估值合理性): 0.20
    df_raw_values['quality_score'] = (
        0.15 * change_norm +
        0.30 * profit_norm +
        0.10 * (1 - turnover_norm) + # 换手率越低，这个值越高
        0.25 * market_cap_norm +
        0.20 * pe_ratio_norm
    )

    y = df_raw_values['quality_score'].values

    # 移除包含 NaN 或无穷大的行
    mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1) & ~np.isnan(y) & ~np.isinf(y)
    X = X[mask]
    y = y[mask]

    if len(X) < 50: # 至少需要更多数据来划分训练集和测试集，并进行Optuna优化
        print("   ❌ 有效训练数据不足，无法训练神经网络。至少需要50个样本。")
        return None, None

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
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=70, show_progress_bar=True) # 增加试验次数
    except Exception as e:
        print(f"   Optuna 优化过程中发生错误: {e}")
        print("   将使用默认或预设参数训练模型。")
        # 如果Optuna失败，使用一个合理的默认配置
        best_params = {
            'n_layers': 2,
            'n_units_l0': 128,
            'n_units_l1': 64,
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate_init': 0.001
        }
    else:
        print("\n   Optuna 优化完成。")
        print(f"   最佳均方误差 (MSE): {study.best_value:.4f}")
        print(f"   最佳超参数: {study.best_params}")
        best_params = study.best_params

    # 使用最佳参数训练最终模型
    print("   使用最佳参数训练最终神经网络模型...")
    hidden_layer_sizes = []
    for i in range(best_params['n_layers']):
        hidden_layer_sizes.append(best_params[f'n_units_l{i}'])

    model = MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        activation=best_params['activation'],
        solver=best_params['solver'],
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        random_state=42,
        max_iter=1500, # 进一步增加最大迭代次数
        early_stopping=True,
        n_iter_no_change=40, # 进一步增加耐心
        tol=1e-5 # 增加容忍度
    )
    model.fit(X_train, y_train)

    # 评估模型
    print("   评估最终模型...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"   最终模型均方误差 (MSE): {mse:.4f}")
    print(f"   最终模型R²分数: {r2:.4f}")

    return model, scaler

def predict_score_with_nn(row, model, scaler, hist_data_map):
    """
    使用训练好的神经网络模型预测股票评分
    """
    features = calculate_all_features(row, hist_data_map)
    # 检查特征中是否有NaN或Inf，如果有，则返回一个默认值或NaN
    if any(pd.isna(f) or np.isinf(f) for f in features):
        return np.nan # 或者一个非常低的默认分数

    features = np.array(features).reshape(1, -1)  # 转换为二维数组
    try:
        features_scaled = scaler.transform(features)
        score = model.predict(features_scaled)[0]
        return score
    except Exception as e:
        # print(f"预测分数时发生错误: {e}, 特征: {features}")
        return np.nan # 预测失败时返回NaN

def generate_strategy_signals(row, nn_score, tech_indicators):
    """
    根据神经网络评分和技术指标生成短期/长期买卖信号。
    nn_score: 神经网络预测的质量评分
    tech_indicators: 该股票的技术指标字典
    """
    signals = []
    current_price = safe_float(row.get('最新'))
    change_percent = safe_float(row.get('涨幅%'))
    turnover_rate = safe_float(row.get('实际换手%'))
    
    # 获取技术指标
    ma5 = tech_indicators.get('MA5', np.nan)
    ma10 = tech_indicators.get('MA10', np.nan)
    ma20 = tech_indicators.get('MA20', np.nan)
    ma60 = tech_indicators.get('MA60', np.nan)
    ma120 = tech_indicators.get('MA120', np.nan)
    ma200 = tech_indicators.get('MA200', np.nan)
    rsi = tech_indicators.get('RSI', np.nan)
    macd_dif = tech_indicators.get('MACD_DIF', np.nan)
    macd_dea = tech_indicators.get('MACD_DEA', np.nan)
    macd_hist = tech_indicators.get('MACD_HIST', np.nan)
    boll_up = tech_indicators.get('BOLL_UP', np.nan)
    boll_low = tech_indicators.get('BOLL_LOW', np.nan)
    vol_ma5 = tech_indicators.get('VOL_MA5', np.nan)
    vol_ma10 = tech_indicators.get('VOL_MA10', np.nan)
    vol_change = tech_indicators.get('VOL_CHANGE', np.nan)

    # 确保所有关键指标非NaN
    if pd.isna(nn_score) or pd.isna(current_price):
        return ["数据不足，无法判断"]

    # --- 买入信号 ---
    # 1. 神经网络高评分
    if nn_score > 0.7: # 阈值可调，0.7表示相对较高的质量评分
        signals.append("NN高评分")

    # 2. 短期买入策略 (偏向动量和超跌反弹)
    if pd.notna(rsi) and pd.notna(macd_hist) and pd.notna(ma5) and pd.notna(ma10) and pd.notna(turnover_rate):
        # RSI超卖反弹
        if rsi < 30 and current_price > ma5:
            signals.append("短期买入: RSI超卖反弹")
        # MACD金叉
        if macd_dif > macd_dea and macd_hist > 0 and macd_dif < 0: # 金叉且在零轴下方
            signals.append("短期买入: MACD金叉 (底部区域)")
        # 价格突破短期均线
        if current_price > ma5 and ma5 > ma10 and change_percent > 2.0: # 价格站上5日线，5日线向上，且有一定涨幅
            signals.append("短期买入: 价格突破短期均线")
        # 放量上涨
        if vol_change > 0.5 and change_percent > 3.0: # 成交量放大50%且涨幅超过3%
            signals.append("短期买入: 放量上涨")
        # 价格接近布林带下轨并反弹
        if pd.notna(boll_low) and current_price > boll_low and (current_price - boll_low) / boll_low < 0.01 and change_percent > 0:
            signals.append("短期买入: 布林带下轨支撑")

    # 3. 长期买入策略 (偏向趋势和价值)
    if pd.notna(ma60) and pd.notna(ma120) and pd.notna(ma200):
        # 长期均线多头排列 (或接近多头排列)
        if ma5 > ma10 > ma20 > ma60 and current_price > ma60:
            signals.append("长期买入: 均线多头排列")
        # 价格站上长期均线
        if current_price > ma60 and ma60 > ma120 and ma120 > ma200:
            signals.append("长期买入: 价格站上长期趋势线")
        # 价值投资考量 (结合NN评分)
        if nn_score > 0.8 and safe_float(row.get('市盈率(动)')) > 0 and safe_float(row.get('市盈率(动)')) < 30: # NN高评分且PE合理
            signals.append("长期买入: 价值与成长兼备")

    # --- 卖出信号 ---
    # 1. 短期卖出策略 (止盈/止损/动量衰竭)
    if pd.notna(rsi) and pd.notna(macd_hist) and pd.notna(ma5) and pd.notna(ma10):
        # RSI超买
        if rsi > 70:
            signals.append("短期卖出: RSI超买")
        # MACD死叉
        if macd_dif < macd_dea and macd_hist < 0 and macd_dif > 0: # 死叉且在零轴上方
            signals.append("短期卖出: MACD死叉 (顶部区域)")
        # 价格跌破短期均线
        if current_price < ma5 and ma5 < ma10 and change_percent < -2.0:
            signals.append("短期卖出: 价格跌破短期均线")
        # 价格跌破布林带中轨或下轨
        if pd.notna(boll_mid) and current_price < boll_mid and change_percent < -1.0:
            signals.append("短期卖出: 跌破布林带中轨")
        if pd.notna(boll_low) and current_price < boll_low:
            signals.append("短期卖出: 跌破布林带下轨")

    # 2. 长期卖出策略 (趋势反转/基本面恶化)
    if pd.notna(ma60) and pd.notna(ma120):
        # 长期均线死叉
        if ma60 < ma120 and current_price < ma60:
            signals.append("长期卖出: 长期均线死叉")
        # 价格跌破长期趋势线
        if current_price < ma60 and ma60 < ma200:
            signals.append("长期卖出: 价格跌破长期趋势线")
        # 基本面恶化 (例如，净利润为负或大幅下降，这里需要更多历史财务数据来判断)
        if safe_float(row.get('归属净利润')) < 0 and safe_float(row.get('总市值')) > 0: # 亏损且非ST股
            signals.append("长期卖出: 基本面恶化 (亏损)")

    if not signals:
        signals.append("无明确信号 (持有/观望)")

    return signals

def perform_association_rule_mining(df):
    """
    使用关联规则挖掘来发现苏氏量化策略条件与高涨幅之间的关系。
    """
    print("\n4. 执行关联规则挖掘...")

    # 准备数据：将特征和目标变量二值化
    data_for_ar = []
    
    # 获取所有股票的历史数据，用于计算技术指标
    today_str = datetime.now().strftime('%Y%m%d')
    start_date_hist = (datetime.now() - timedelta(days=300)).strftime('%Y%m%d')
    
    hist_data_map = {}
    for symbol in df['原始代码'].unique():
        hist_data_map[symbol] = calculate_technical_indicators(
            get_stock_history_data(symbol, start_date_hist, today_str)
        )

    for _, row in df.iterrows():
        features_list = calculate_all_features(row, hist_data_map)
        items = []

        # 苏氏量化策略特征
        if features_list[0] == 1: items.append("F_价格位置_满足")
        else: items.append("F_价格位置_不满足")
        if features_list[1] == 1: items.append("G_涨幅位置_满足")
        else: items.append("G_涨幅位置_不满足")

        # 基本面特征
        if features_list[2] >= 0.3: items.append("H_净利润_高") # 0.3亿
        else: items.append("H_净利润_低")
        if features_list[3] <= 20: items.append("I_换手率_低") # 20%
        else: items.append("I_换手率_高")
        if features_list[4] >= 300: items.append("J_市值_大") # 300亿
        else: items.append("J_市值_小")
        if features_list[5] > 0 and features_list[5] < 50: items.append("K_市盈率_合理") # 0-50
        else: items.append("K_市盈率_不合理")

        # 技术指标特征 (二值化)
        current_price = safe_float(row.get('最新'))
        ma20 = features_list[8] # MA20
        ma60 = features_list[9] # MA60
        rsi = features_list[11] # RSI
        macd_hist = features_list[14] # MACD_HIST

        if pd.notna(current_price) and pd.notna(ma20) and current_price > ma20: items.append("技术_价格高于MA20")
        else: items.append("技术_价格低于MA20")
        if pd.notna(current_price) and pd.notna(ma60) and current_price > ma60: items.append("技术_价格高于MA60")
        else: items.append("技术_价格低于MA60")
        if pd.notna(rsi) and rsi < 30: items.append("技术_RSI超卖")
        if pd.notna(rsi) and rsi > 70: items.append("技术_RSI超买")
        if pd.notna(macd_hist) and macd_hist > 0: items.append("技术_MACD金叉")
        else: items.append("技术_MACD死叉")

        # 目标变量：高涨幅 (例如，涨幅 > 2%)
        change = safe_float(row.get('涨幅%'))
        if pd.notna(change) and change > 2.0: # 可以调整这个阈值
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
    frequent_itemsets = apriori(df_ar, min_support=0.005, use_colnames=True) # 降低min_support以发现更多规则
    if frequent_itemsets.empty:
        print("   ⚠️ 未找到频繁项集，请尝试降低 min_support。")
        return

    # 生成关联规则
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2) # 提高min_threshold以获取更强的关联
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
        for i, rule in high_return_rules.head(15).iterrows(): # 只显示前15条
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
    print("动态选股系统 - 实时计算版 (集成神经网络与关联规则)")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 创建输出目录
    os.makedirs('输出数据', exist_ok=True)

    # ========== 第一步：获取数据 ==========
    print("\n1. 获取A股数据...")

    df = pd.DataFrame()
    # 先尝试获取实时数据
    try:
        print("   尝试获取实时数据...")
        df_realtime = ak.stock_zh_a_spot_em()
        print(f"   ✅ 成功获取 {len(df_realtime)} 只股票的实时数据")

        # 统一列名
        df_realtime.rename(columns={
            '最新价': '最新',
            '涨跌幅': '涨幅%',
            '换手率': '实际换手%',
            '市盈率-动态': '市盈率(动)' # 确保列名一致
        }, inplace=True)

        # 保存原始代码
        df_realtime['原始代码'] = df_realtime['代码'].copy()

        # 格式化代码
        df_realtime['代码'] = df_realtime['代码'].apply(lambda x: f'= "{str(x)}"')

        # 确保所有关键列存在，并初始化为None或默认值
        required_cols = ['代码', '名称', '最新', '涨幅%', '最高', '最低', '实际换手%',
                         '所属行业', '20日均价', '60日均价', '市盈率(动)', '总市值',
                         '归属净利润', '昨收', '开盘', '原始代码']
        for col in required_cols:
            if col not in df_realtime.columns:
                df_realtime[col] = np.nan # 使用NaN方便后续处理

        df = df_realtime

    except Exception as e:
        print(f"   ❌ 实时获取失败: {e}")
        print("   使用参考数据作为备选...")

        # 使用参考数据
        try:
            df_ref = pd.read_csv('参考数据/Table.xls', sep='\t', encoding='gbk', dtype=str)
            print(f"   ✅ 从参考文件加载了 {len(df_ref)} 条数据")
            df_ref['原始代码'] = df_ref['代码'].str.replace('= "', '').str.replace('"', '')
            df = df_ref
        except Exception as e2:
            print(f"   ❌ 无法加载参考数据: {e2}")
            return

    # 尝试补充均线和财务数据 (如果实时数据缺失)
    try:
        ref_df_path = '参考数据/Table.xls'
        if os.path.exists(ref_df_path):
            ref_df = pd.read_csv(ref_df_path, sep='\t', encoding='gbk', dtype=str)
            ref_map = {}
            for _, row in ref_df.iterrows():
                code = str(row['代码']).replace('= "', '').replace('"', '')
                ref_map[code] = row.to_dict()

            # 合并参考数据
            merged_count = 0
            for i, row in df.iterrows():
                code = row.get('原始代码')
                if code and code in ref_map:
                    ref = ref_map[code]
                    # 补充缺失的数据
                    for col in ['20日均价', '60日均价', '所属行业', '归属净利润', '总市值', '市盈率(动)']:
                        if col in ref and pd.isna(df.loc[i, col]): # 只补充NaN的值
                            df.loc[i, col] = ref[col]
                    merged_count += 1
            print(f"   ✅ 补充了 {merged_count} 条参考数据")
        else:
            print("   ⚠️ 未找到参考数据文件 '参考数据/Table.xls'，无法补充数据。")
    except Exception as e:
        print(f"   ⚠️ 补充参考数据时发生错误: {e}")

    # 统一数据格式
    for col in ['最新', '最高', '最低', '开盘', '昨收', '涨幅%', '实际换手%', '20日均价', '60日均价', '市盈率(动)', '总市值', '归属净利润']:
        df[col] = df[col].apply(safe_float)

    # 添加序号
    df['序'] = range(1, len(df) + 1)
    df['Unnamed: 16'] = '' # 保持与原文件格式一致

    # 选择输出列
    output_columns = [
        '序', '代码', '名称', '最新', '涨幅%', '最高', '最低',
        '实际换手%', '所属行业', '20日均价', '60日均价',
        '市盈率(动)', '总市值', '归属净利润', '昨收', '开盘', 'Unnamed: 16'
    ]

    # 确保所有输出列都存在，并填充默认值
    for col in output_columns:
        if col not in df.columns:
            df[col] = np.nan if col not in ['代码', '名称', 'Unnamed: 16'] else ' --'

    # 格式化输出到CSV的数值列
    df_for_output_csv = df[output_columns].copy() # 复制一份用于CSV输出
    for col in ['最新', '涨幅%', '最高', '最低', '实际换手%', '20日均价', '60日均价', '市盈率(动)', '总市值', '归属净利润', '昨收', '开盘']:
        df_for_output_csv[col] = df_for_output_csv[col].apply(lambda x: f" {x:.2f}" if pd.notna(x) else " --")

    # 格式化代码和名称
    df_for_output_csv['代码'] = df_for_output_csv['代码'].apply(lambda x: f'= "{str(x)}"' if not str(x).startswith('=') else x)
    df_for_output_csv['名称'] = df_for_output_csv['名称'].apply(lambda x: f" {x}" if not str(x).startswith(' ') else x)

    # 保存A股数据
    output_file1 = '输出数据/A股数据.csv'
    df_for_output_csv.to_csv(output_file1, index=False, encoding='utf-8-sig')
    print(f"\n✅ A股数据已保存: {output_file1}")
    print(f"   共 {len(df_for_output_csv)} 只股票")

    # ========== 第二步：训练神经网络 ==========
    print("\n2. 训练神经网络模型...")
    # 传入原始数值的df副本，避免格式化影响训练
    df_for_training = df.copy()
    for col in ['最新', '涨幅%', '最高', '最低', '开盘', '昨收', '实际换手%', '20日均价', '60日均价', '市盈率(动)', '总市值', '归属净利润']:
        df_for_training[col] = df_for_training[col].apply(safe_float)

    model, scaler = train_neural_network(df_for_training)

    if model is None:
        print("   ❌ 神经网络训练失败，无法进行后续筛选。")
        return

    # ========== 第三步：动态筛选优质股票并生成策略信号 ==========
    print("\n3. 动态筛选优质股票 (基于神经网络评分和策略信号)...")

    quality_stocks = []
    
    # 重新加载原始数值的df，因为上面为了输出csv已经格式化了
    df_for_scoring = df.copy()
    for col in ['最新', '涨幅%', '最高', '最低', '实际换手%', '20日均价', '60日均价', '市盈率(动)', '总市值', '归属净利润', '昨收', '开盘']:
        df_for_scoring[col] = df_for_scoring[col].apply(safe_float)
    df_for_scoring['原始代码'] = df_for_scoring['代码'].apply(lambda x: str(x).replace('= "', '').replace('"', ''))

    # 预先获取所有股票的历史数据，避免循环中重复获取
    today_str = datetime.now().strftime('%Y%m%d')
    start_date_hist = (datetime.now() - timedelta(days=300)).strftime('%Y%m%d') # 足够长的时间来计算200日均线
    
    all_tech_indicators_map = {}
    print("   正在获取所有股票历史数据并计算技术指标...")
    for symbol in df_for_scoring['原始代码'].unique():
        all_tech_indicators_map[symbol] = calculate_technical_indicators(
            get_stock_history_data(symbol, start_date_hist, today_str)
        )
    print("   ✅ 技术指标计算完成。")

    for idx, row in df_for_scoring.iterrows():
        symbol = str(row['原始代码']).strip()
        tech_indicators = all_tech_indicators_map.get(symbol, {}) # 获取该股票的技术指标

        nn_score = predict_score_with_nn(row, model, scaler, all_tech_indicators_map)
        signals = generate_strategy_signals(row, nn_score, tech_indicators)
        
        if pd.notna(nn_score): # 确保分数有效
            quality_stocks.append({
                '代码': symbol,
                '名称': str(row['名称']).strip(),
                '行业': str(row['所属行业']).strip(),
                '优质率': nn_score,
                '今日涨幅': f"{safe_float(row['涨幅%']):.2f}%" if pd.notna(safe_float(row['涨幅%'])) else "--",
                '策略信号': ", ".join(signals)
            })

    # 按优质率降序排序
    quality_stocks = sorted(quality_stocks, key=lambda x: (x['优质率'], x['代码']), reverse=True)

    # 确定筛选阈值：取前N个，或者根据分数分布动态调整
    display_count = 20 # 默认显示前20个
    if len(quality_stocks) > display_count:
        threshold = quality_stocks[display_count-1]['优质率']
        quality_stocks_filtered = quality_stocks[:display_count]
    elif len(quality_stocks) > 0:
        threshold = quality_stocks[-1]['优质率'] # 所有股票的最低分
        quality_stocks_filtered = quality_stocks
    else:
        threshold = 0.0
        quality_stocks_filtered = []

    # 保存优质股票和策略信号
    output_file2 = '输出数据/优质股票与策略信号.txt'
    with open(output_file2, 'w', encoding='utf-8') as f:
        f.write("苏氏量化策略 - 优质股票筛选结果 (神经网络评分与策略信号)\n")
        f.write(f"筛选时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"最低优质率阈值 (基于前{display_count}名或全部): {threshold:.4f}\n")
        f.write(f"优质股票数量: {len(quality_stocks_filtered)}\n")
        f.write("="*80 + "\n\n")

        for stock in quality_stocks_filtered:
            f.write(f"股票代码: {stock['代码']}\n")
            f.write(f"股票名称: {stock['名称']}\n")
            f.write(f"所属行业: {stock['行业']}\n")
            f.write(f"优质率 (NN评分): {stock['优质率']:.4f}\n")
            f.write(f"今日涨幅: {stock['今日涨幅']}\n")
            f.write(f"策略信号: {stock['策略信号']}\n")
            f.write("-"*40 + "\n")

    print(f"\n✅ 优质股票与策略信号已保存: {output_file2}")
    print(f"   找到 {len(quality_stocks_filtered)} 只优质股票（最低优质率={threshold:.4f}）")

    if len(quality_stocks_filtered) > 0:
        print(f"\n🎯 今日优质股票列表及策略信号 (前{len(quality_stocks_filtered)}名)：")
        print("="*100)
        print(f"{'股票代码':<10} {'股票名称':<12} {'涨幅':<8} {'优质率':<10} {'所属行业':<15} {'策略信号':<40}")
        print("-"*100)
        for stock in quality_stocks_filtered:
            print(f"{stock['代码']:<10} {stock['名称']:<12} {stock['今日涨幅']:<8} {stock['优质率']:.4f}   {stock['行业']:<15} {stock['策略信号']:<40}")
    else:
        print("\n⚠️ 今日没有找到符合条件的优质股票")
        print("   可能原因：")
        print("   1. 市场整体表现不佳，涨幅不足")
        print("   2. 数据获取不完整或质量不佳")
        print("   3. 神经网络模型需要更多数据或优化")
        print("   4. 策略信号条件过于严格")

    # ========== 第四步：关联规则挖掘 ==========
    # 在这里调用关联规则挖掘函数
    perform_association_rule_mining(df_for_scoring.copy()) # 传入原始数值的df副本

    print("\n" + "="*60)
    print("✅ 程序执行完成！")
    print("="*60)


if __name__ == "__main__":
    main()
