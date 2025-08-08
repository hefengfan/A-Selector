#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态选股系统 - 实时计算版 (集成机器学习与多策略)
基于苏氏量化策略的真实计算逻辑
集成更精密的机器学习模型 (XGBoost) 进行精准评分和信号生成
新增：
1. 更严格的数据清洗和预处理，处理缺失值、异常值。
2. 特征选择，使用特征重要性分析选择最相关的特征。
3. 更稳定的模型训练，增加交叉验证的折数。
4. 更精细的信号生成，结合多种信号，并引入置信度评估。
5. 更清晰的结果展示，提供更详细的交易信号和回测报告。
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# 导入机器学习相关库
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import optuna

# 导入技术分析库
import ta

# 导入 mlxtend 进行关联规则挖掘
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 清除代理设置
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['ALL_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

# ==================== 配置参数 ====================
# 历史数据获取范围 (用于模型训练和指标计算)
HISTORY_DAYS = 252 * 2 # 约2年交易日数据
# 预测未来收益的天数 (短期/长期)
SHORT_TERM_FUTURE_DAYS = 5  # 预测未来5天收益
LONG_TERM_FUTURE_DAYS = 20  # 预测未来20天收益

# 信号生成阈值
BUY_SCORE_THRESHOLD_SHORT = 0.02 # 短期预测收益率高于此值则买入
BUY_SCORE_THRESHOLD_LONG = 0.05  # 长期预测收益率高于此值则买入

# 关联规则挖掘参数
AR_MIN_SUPPORT = 0.01
AR_MIN_THRESHOLD = 1.1

# 交叉验证折数
CV_FOLDS = 5

# ==================== 辅助函数 ====================

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
            return float(s.replace('亿', '')) * 100000000 # 1亿 = 10000万
        if '万亿' in s:
            return float(s.replace('万亿', '')) * 1000000000000 # 1万亿 = 10000亿
        if '万' in s:
            return float(s.replace('万', '')) * 10000 # 1万
        return float(s)
    except ValueError:
        return np.nan

def get_stock_data(symbol, start_date, end_date):
    """
    获取指定股票的历史日K线数据。
    """
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        # 确保数值列为float
        for col in ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        # print(f"获取 {symbol} 历史数据失败: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """
    计算多种技术指标。
    输入: 包含 '开盘', '收盘', '最高', '最低', '成交量' 的DataFrame
    输出: 包含技术指标的DataFrame
    """
    if df.empty:
        return df

    # 移动平均线
    df['SMA_5'] = ta.trend.sma_indicator(df['收盘'], window=5)
    df['SMA_10'] = ta.trend.sma_indicator(df['收盘'], window=10)
    df['SMA_20'] = ta.trend.sma_indicator(df['收盘'], window=20)
    df['EMA_5'] = ta.trend.ema_indicator(df['收盘'], window=5)
    df['EMA_10'] = ta.trend.ema_indicator(df['收盘'], window=10)
    df['EMA_20'] = ta.trend.ema_indicator(df['收盘'], window=20)

    # MACD
    macd = ta.trend.MACD(df['收盘'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()

    # RSI
    df['RSI'] = ta.momentum.rsi(df['收盘'], window=14)

    # KDJ (Stochastic Oscillator)
    stoch = ta.momentum.StochasticOscillator(df['最高'], df['最低'], df['收盘'])
    df['K'] = stoch.stoch()
    df['D'] = stoch.stoch_signal()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['收盘'])
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['BB_Width'] = bollinger.bollinger_wband()

    # ATR (Average True Range)
    df['ATR'] = ta.volatility.average_true_range(df['最高'], df['最低'], df['收盘'], window=14)

    # 成交量移动平均
    df['Volume_SMA_5'] = ta.volume.volume_sma_indicator(df['成交量'], window=5)
    df['Volume_SMA_10'] = ta.volume.volume_sma_indicator(df['成交量'], window=10)

    # 价格变动率
    df['Daily_Return'] = df['收盘'].pct_change()
    df['Log_Return'] = np.log(df['收盘'] / df['收盘'].shift(1))

    # 价格位置 (苏氏策略)
    df['Price_Pos_F'] = 0
    df.loc[(df['最低'] / df['SMA_60'] >= 0.85) & (df['最低'] / df['SMA_60'] <= 1.15), 'Price_Pos_F'] = 1
    df.loc[(df['收盘'] / df['SMA_20'] >= 0.90) & (df['收盘'] / df['SMA_20'] <= 1.10), 'Price_Pos_F'] = 1

    # 涨幅和价格位置 (苏氏策略)
    df['Price_Pos_G'] = 0
    df.loc[(df['涨跌幅'] >= 5.0) & ((df['收盘'] >= (df['最高'] - (df['最高'] - df['最低']) * 0.30)) | (df['最高'] == df['最低'])), 'Price_Pos_G'] = 1

    return df

def prepare_data_for_ml(df_all_stocks, future_days):
    """
    准备用于机器学习的数据集，包括特征和目标变量。
    目标变量为未来 N 日的收益率。
    """
    features = []
    targets = []
    stock_codes = []
    dates = []

    # 确保所有数值列都是float类型
    numeric_cols = ['最新', '涨幅%', '最高', '最低', '实际换手%', '20日均价', '60日均价',
                    '市盈率(动)', '总市值', '归属净利润', '昨收', '开盘',
                    'SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_20',
                    'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'K', 'D', 'J',
                    'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR', 'Volume_SMA_5',
                    'Volume_SMA_10', 'Daily_Return', 'Log_Return', 'Price_Pos_F', 'Price_Pos_G']

    for col in numeric_cols:
        if col in df_all_stocks.columns:
            df_all_stocks[col] = df_all_stocks[col].apply(safe_float)

    # 遍历每只股票，计算其特征和未来收益
    for stock_code in df_all_stocks['原始代码'].unique():
        stock_df = df_all_stocks[df_all_stocks['原始代码'] == stock_code].sort_index() # 确保按日期排序

        # 计算未来收益率作为目标变量
        # 未来 N 日的收盘价 / 当日收盘价 - 1
        stock_df['future_return'] = stock_df['收盘'].shift(-future_days) / stock_df['收盘'] - 1

        # 提取特征
        for i in range(len(stock_df)):
            row = stock_df.iloc[i]
            # 确保目标变量存在且不是NaN
            if pd.notna(row['future_return']):
                # 提取所有数值特征
                current_features = []
                for col in numeric_cols:
                    if col in row.index:
                        current_features.append(row[col])
                    else:
                        current_features.append(np.nan) # 如果列不存在，则填充NaN

                # 过滤掉特征中包含NaN或Inf的行
                if not any(pd.isna(f) or np.isinf(f) for f in current_features):
                    features.append(current_features)
                    targets.append(row['future_return'])
                    stock_codes.append(stock_code)
                    dates.append(row.name) # 记录日期

    X = np.array(features)
    y = np.array(targets)

    # 移除包含 NaN 或无穷大的行
    mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1) & ~np.isnan(y) & ~np.isinf(y)
    X = X[mask]
    y = y[mask]
    stock_codes_filtered = np.array(stock_codes)[mask]
    dates_filtered = np.array(dates)[mask]

    # 获取特征列名，用于后续的特征重要性分析或调试
    feature_names = numeric_cols

    return X, y, stock_codes_filtered, dates_filtered, feature_names

def objective_xgb(trial, X_train, y_train, X_test, y_test):
    """
    Optuna 优化 XGBoostRegressor 的目标函数
    """
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-2, 1e3),
        'seed': 42,
        'n_jobs': -1,
    }

    if param['booster'] == 'gbtree':
        param['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
        param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
        param['max_depth'] = trial.suggest_int('max_depth', 3, 9)
    elif param['booster'] == 'dart':
        param['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
        param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        param['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
        param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)

    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              early_stopping_rounds=50, # 提前停止
              verbose=False)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

def train_ml_model(df_all_stocks, future_days):
    """
    训练机器学习模型，预测股票未来收益率。
    """
    print(f"\n   准备训练数据 (预测未来 {future_days} 天收益率)...")
    X, y, _, _, feature_names = prepare_data_for_ml(df_all_stocks.copy(), future_days)

    if len(X) < 50: # 至少需要一些数据来划分训练集和测试集
        print(f"   ❌ 有效训练数据不足 ({len(X)} 条)，无法训练模型。")
        return None, None, None, None

    print(f"   有效训练样本数: {len(X)}")

    # 数据预处理
    print("   数据预处理 (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 特征选择 (基于特征重要性)
    print("   特征选择 (基于特征重要性)...")
    model_for_importance = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model_for_importance.fit(X_scaled, y)
    importances = model_for_importance.feature_importances_
    # 选择重要性大于平均值的特征
    selected_features_indices = np.where(importances > np.mean(importances))[0]
    X_scaled_selected = X_scaled[:, selected_features_indices]
    selected_feature_names = [feature_names[i] for i in selected_features_indices]
    print(f"   选择了 {len(selected_feature_names)} 个重要特征。")

    # 交叉验证
    print(f"   进行 {CV_FOLDS} 折交叉验证...")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_scores = []

    best_model = None
    best_score = float('inf') # 初始化为正无穷

    for fold, (train_index, val_index) in enumerate(kf.split(X_scaled_selected, y)):
        print(f"     Fold {fold+1}/{CV_FOLDS}...")
        X_train, X_val = X_scaled_selected[train_index], X_scaled_selected[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Optuna 超参数优化
        print("       启动 Optuna 超参数优化 (可能需要一些时间)...")
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        try:
            study.optimize(lambda trial: objective_xgb(trial, X_train, y_train, X_val, y_val), n_trials=20, show_progress_bar=False)
        except Exception as e:
            print(f"       Optuna 优化过程中发生错误: {e}")
            print("       将使用默认或预设参数训练模型。")
            best_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': 'gbtree',
                'lambda': 1,
                'alpha': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'eta': 0.1,
                'max_depth': 6,
                'seed': 42,
                'n_jobs': -1,
            }
        else:
            print("       Optuna 优化完成。")
            best_params = study.best_params

        # 使用最佳参数训练模型
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=20,
                  verbose=False)

        # 评估模型
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        cv_scores.append(rmse)

        if rmse < best_score:
            best_score = rmse
            best_model = model

    print(f"   交叉验证完成。平均 RMSE: {np.mean(cv_scores):.4f}")

    return best_model, scaler, selected_feature_names, selected_features_indices

def predict_future_return(row, model, scaler, feature_names, selected_features_indices):
    """
    使用训练好的模型预测股票未来收益率。
    """
    # 提取当前行的特征
    current_features = []
    for col in feature_names:
        if col in row.index:
            current_features.append(safe_float(row[col]))
        else:
            current_features.append(np.nan)

    # 检查特征中是否有NaN或Inf
    if any(pd.isna(f) or np.isinf(f) for f in current_features):
        return np.nan

    features_array = np.array(current_features).reshape(1, -1)
    try:
        features_scaled = scaler.transform(features_array)
        # 只选择训练模型时使用的特征
        features_scaled_selected = features_scaled[:, selected_features_indices]
        predicted_return = model.predict(features_scaled_selected)[0]
        return predicted_return
    except Exception as e:
        # print(f"预测未来收益时发生错误: {e}, 特征: {current_features}")
        return np.nan

def generate_trading_signals(df_current_data, short_term_model, short_term_scaler, short_term_features, short_term_indices,
                             long_term_model, long_term_scaler, long_term_features, long_term_indices):
    """
    根据模型预测生成买入/卖出信号。
    """
    print("\n3. 生成交易信号...")
    signals = []

    # 确保所有数值列都是float类型
    numeric_cols = ['最新', '涨幅%', '最高', '最低', '实际换手%', '20日均价', '60日均价',
                    '市盈率(动)', '总市值', '归属净利润', '昨收', '开盘']
    for col in numeric_cols:
        if col in df_current_data.columns:
            df_current_data[col] = df_current_data[col].apply(safe_float)

    for idx, row in df_current_data.iterrows():
        stock_code = str(row['原始代码']).strip()
        stock_name = str(row['名称']).strip()
        industry = str(row['所属行业']).strip()
        current_price = safe_float(row['最新'])
        current_change = safe_float(row['涨幅%'])

        short_pred_return = np.nan
        long_pred_return = np.nan
        signal_short = "持有"
        signal_long = "持有"
        overall_signal = "持有"
        confidence = 0.0 # 信号置信度

        # 预测短期收益
        if short_term_model and short_term_scaler and short_term_features:
            short_pred_return = predict_future_return(row, short_term_model, short_term_scaler, short_term_features, short_term_indices)
            if pd.notna(short_pred_return):
                if short_pred_return >= BUY_SCORE_THRESHOLD_SHORT:
                    signal_short = "短期买入"
                    confidence += 0.5 # 增加置信度
                elif short_pred_return < -0.02: # 假设低于-2%则建议卖出
                    signal_short = "短期卖出"
                    confidence += 0.5

        # 预测长期收益
        if long_term_model and long_term_scaler and long_term_features:
            long_pred_return = predict_future_return(row, long_term_model, long_term_scaler, long_term_features, long_term_indices)
            if pd.notna(long_pred_return):
                if long_pred_return >= BUY_SCORE_THRESHOLD_LONG:
                    signal_long = "长期买入"
                    confidence += 0.5
                elif long_pred_return < -0.05: # 假设低于-5%则建议卖出
                    signal_long = "长期卖出"
                    confidence += 0.5

        # 综合信号判断 (可根据实际策略调整)
        if signal_short == "短期买入" and signal_long == "长期买入":
            overall_signal = "强烈买入"
            confidence = 1.0
        elif signal_short == "短期买入" and signal_long == "持有":
            overall_signal = "短期买入"
            confidence = 0.75
        elif signal_long == "长期买入" and signal_short == "持有":
            overall_signal = "长期买入"
            confidence = 0.75
        elif signal_short == "短期卖出" or signal_long == "长期卖出":
            overall_signal = "卖出"
            confidence = 0.75

        signals.append({
            '代码': stock_code,
            '名称': stock_name,
            '行业': industry,
            '最新价': f"{current_price:.2f}" if pd.notna(current_price) else "--",
            '今日涨幅': f"{current_change:.2f}%" if pd.notna(current_change) else "--",
            f'预测{SHORT_TERM_FUTURE_DAYS}日收益': f"{short_pred_return*100:.2f}%" if pd.notna(short_pred_return) else "--",
            f'预测{LONG_TERM_FUTURE_DAYS}日收益': f"{long_pred_return*100:.2f}%" if pd.notna(long_pred_return) else "--",
            '短期信号': signal_short,
            '长期信号': signal_long,
            '综合信号': overall_signal,
            '信号置信度': f"{confidence:.2f}"
        })

    signals_df = pd.DataFrame(signals)
    # 排序，优先显示强烈买入，然后是长期买入，短期买入，再按预测收益排序
    signal_order = {"强烈买入": 4, "长期买入": 3, "短期买入": 2, "持有": 1, "卖出": 0}
    signals_df['signal_rank'] = signals_df['综合信号'].map(signal_order)

    # 将预测收益转换为数值以便排序
    signals_df[f'预测{SHORT_TERM_FUTURE_DAYS}日收益_num'] = signals_df[f'预测{SHORT_TERM_FUTURE_DAYS}日收益'].str.replace('%', '').apply(safe_float) / 100
    signals_df[f'预测{LONG_TERM_FUTURE_DAYS}日收益_num'] = signals_df[f'预测{LONG_TERM_FUTURE_DAYS}日收益'].str.replace('%', '').apply(safe_float) / 100

    signals_df = signals_df.sort_values(by=['signal_rank', f'预测{LONG_TERM_FUTURE_DAYS}日收益_num', f'预测{SHORT_TERM_FUTURE_DAYS}日收益_num'], ascending=[False, False, False])

    # 删除辅助列
    signals_df = signals_df.drop(columns=['signal_rank', f'预测{SHORT_TERM_FUTURE_DAYS}日收益_num', f'预测{LONG_TERM_FUTURE_DAYS}日收益_num'])

    return signals_df

def perform_association_rule_mining(df):
    """
    使用关联规则挖掘来发现苏氏量化策略条件与高涨幅之间的关系。
    """
    print("\n4. 执行关联规则挖掘...")

    # 准备数据：将特征和目标变量二值化
    data_for_ar = []
    # 确保所有数值列都是float类型
    numeric_cols = ['最新', '涨幅%', '最高', '最低', '实际换手%', '20日均价', '60日均价',
                    '市盈率(动)', '总市值', '归属净利润', '昨收', '开盘',
                    'SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_20',
                    'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'K', 'D', 'J',
                    'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR', 'Volume_SMA_5',
                    'Volume_SMA_10', 'Daily_Return', 'Log_Return', 'Price_Pos_F', 'Price_Pos_G']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)

    for _, row in df.iterrows():
        items = []

        # 苏氏策略特征
        f_condition = safe_float(row.get('Price_Pos_F'))
        g_condition = safe_float(row.get('Price_Pos_G'))
        profit = safe_float(row.get('归属净利润'))
        turnover = safe_float(row.get('实际换手%'))
        cap = safe_float(row.get('总市值'))
        pe_ratio = safe_float(row.get('市盈率(动)'))

        if pd.notna(f_condition):
            items.append("F_价格位置_满足" if f_condition == 1 else "F_价格位置_不满足")
        if pd.notna(g_condition):
            items.append("G_涨幅位置_满足" if g_condition == 1 else "G_涨幅位置_不满足")
        if pd.notna(profit):
            items.append("H_净利润_高" if profit >= 0.3 else "H_净利润_低") # 0.3亿
        if pd.notna(turnover):
            items.append("I_换手率_低" if turnover <= 20 else "I_换手率_高")
        if pd.notna(cap):
            items.append("J_市值_大" if cap >= 300 else "J_市值_小") # 300亿
        if pd.notna(pe_ratio):
            items.append("K_市盈率_低" if pe_ratio > 0 and pe_ratio <= 30 else "K_市盈率_高") # 0-30为低

        # 目标变量：高涨幅 (例如，涨幅 > 2%)
        change = safe_float(row.get('涨幅%'))
        if pd.notna(change) and change > 2.0: # 可以调整这个阈值
            items.append("高涨幅")
        else:
            items.append("低涨幅")

        if items: # 确保有有效项
            data_for_ar.append(items)

    if not data_for_ar:
        print("   ❌ 没有足够的数据进行关联规则挖掘。")
        return

    te = TransactionEncoder()
    te_ary = te.fit(data_for_ar).transform(data_for_ar)
    df_ar = pd.DataFrame(te_ary, columns=te.columns_)

    # 查找频繁项集
    frequent_itemsets = apriori(df_ar, min_support=AR_MIN_SUPPORT, use_colnames=True)
    if frequent_itemsets.empty:
        print("   ⚠️ 未找到频繁项集，请尝试降低 min_support。")
        return

    # 生成关联规则
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=AR_MIN_THRESHOLD)

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

def simplified_backtest(signals_df, initial_capital=100000):
    """
    简化的回测功能，模拟根据信号进行交易。
    注意：这是一个非常简化的版本，不考虑滑点、佣金、资金管理等复杂因素。
    仅用于初步评估策略的潜在方向。
    """
    print("\n5. 简化的策略回测 (仅供参考，不含真实交易细节)...")

    # 假设我们只关注“强烈买入”和“卖出”信号
    buy_signals = signals_df[signals_df['综合信号'] == '强烈买入']
    sell_signals = signals_df[signals_df['综合信号'] == '卖出']

    if buy_signals.empty and sell_signals.empty:
        print("   没有生成买入或卖出信号，无法进行回测。")
        return

    print(f"   初始资金: {initial_capital:.2f} 元")
    current_capital = initial_capital
    positions = {} # {股票代码: 持有数量}
    trade_log = []

    # 模拟交易 (非常粗略的逻辑)
    # 假设我们只在信号生成当天进行交易，并且只买入强烈买入的股票，卖出卖出信号的股票
    # 并且假设我们能以最新价成交

    # 卖出操作 (先处理卖出，释放资金)
    for _, row in sell_signals.iterrows():
        code = row['代码']
        if code in positions and positions[code] > 0:
            latest_price = safe_float(row['最新价'])
            if pd.notna(latest_price) and latest_price > 0:
                profit_loss = (latest_price - positions[code]['avg_price']) * positions[code]['quantity']
                current_capital += latest_price * positions[code]['quantity']
                trade_log.append(f"   卖出 {code} ({row['名称']}): 数量 {positions[code]['quantity']}, 价格 {latest_price:.2f}, 盈亏 {profit_loss:.2f}")
                del positions[code]
            else:
                trade_log.append(f"   卖出 {code} ({row['名称']}): 价格数据缺失，无法执行。")

    # 买入操作 (用剩余资金买入强烈买入的股票)
    if not buy_signals.empty:
        # 平均分配资金给所有强烈买入的股票
        num_buy_stocks = len(buy_signals)
        if num_buy_stocks > 0:
            capital_per_stock = current_capital / num_buy_stocks
            for _, row in buy_signals.iterrows():
                code = row['代码']
                latest_price = safe_float(row['最新价'])
                if pd.notna(latest_price) and latest_price > 0:
                    quantity = int(capital_per_stock / latest_price / 100) * 100 # 购买100股的整数倍
                    if quantity > 0:
                        cost = quantity * latest_price
                        current_capital -= cost
                        positions[code] = {'quantity': quantity, 'avg_price': latest_price}
                        trade_log.append(f"   买入 {code} ({row['名称']}): 数量 {quantity}, 价格 {latest_price:.2f}, 成本 {cost:.2f}")
                else:
                    trade_log.append(f"   买入 {code} ({row['名称']}): 价格数据缺失，无法执行。")

    # 计算当前总资产
    current_portfolio_value = current_capital
    for code, pos in positions.items():
        # 尝试获取当前持仓股票的最新价格 (这里简化为使用信号DF中的最新价)
        latest_price_in_df = safe_float(signals_df[signals_df['代码'] == code]['最新价'].iloc[0])
        if pd.notna(latest_price_in_df):
            current_portfolio_value += pos['quantity'] * latest_price_in_df
        else:
            # 如果最新价缺失，则用买入时的平均价估算
            current_portfolio_value += pos['quantity'] * pos['avg_price']


    total_return = (current_portfolio_value - initial_capital) / initial_capital * 100

    print("\n   --- 交易日志 ---")
    for log in trade_log:
        print(log)
    print("\n   --- 回测结果 ---")
    print(f"   期末总资产: {current_portfolio_value:.2f} 元")
    print(f"   总收益率: {total_return:.2f}%")
    print(f"   当前持仓股票数量: {len(positions)}")
    if positions:
        print("   当前持仓明细:")
        for code, pos in positions.items():
            print(f"     - {code}: 数量 {pos['quantity']}, 均价 {pos['avg_price']:.2f}")
    print("   回测结束。")


# ==================== 主程序 ====================

def main():
    """主程序"""
    print("\n" + "="*60)
    print("动态选股系统 - 实时计算版 (集成机器学习与多策略)")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 创建输出目录
    os.makedirs('输出数据', exist_ok=True)

    # ========== 第一步：获取并整合数据 ==========
    print("\n1. 获取A股数据 (实时 + 历史)...")

    # 获取所有A股实时数据
    df_realtime = pd.DataFrame()
    try:
        print("   尝试获取实时数据...")
        df_realtime = ak.stock_zh_a_spot_em()
        print(f"   ✅ 成功获取 {len(df_realtime)} 只股票的实时数据")

        df_realtime.rename(columns={
            '最新价': '最新', '涨跌幅': '涨幅%', '换手率': '实际换手%',
            '市盈率-动态': '市盈率(动)', '总市值': '总市值', '归属净利润': '归属净利润'
        }, inplace=True)
        df_realtime['原始代码'] = df_realtime['代码'].copy()
        df_realtime['代码'] = df_realtime['代码'].apply(lambda x: f'= "{str(x)}"')

        # 确保所有关键列存在
        required_cols = ['代码', '名称', '最新', '涨幅%', '最高', '最低', '实际换手%',
                         '所属行业', '市盈率(动)', '总市值', '归属净利润', '昨收', '开盘', '原始代码']
        for col in required_cols:
            if col not in df_realtime.columns:
                df_realtime[col] = np.nan

    except Exception as e:
        print(f"   ❌ 实时获取失败: {e}")
        print("   将尝试从参考数据补充。")
        df_realtime = pd.DataFrame(columns=required_cols) # 创建空DataFrame以避免后续错误

    # 获取历史数据并计算指标
    all_historical_data = []
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime('%Y%m%d')

    print(f"   获取历史日K线数据 ({start_date} 至 {end_date})...")
    # 使用实时数据中的股票代码列表
    stock_codes_to_fetch = df_realtime['原始代码'].tolist()
    if not stock_codes_to_fetch: # 如果实时数据获取失败，尝试从参考数据获取代码
        try:
            ref_df_path = '参考数据/Table.xls'
            if os.path.exists(ref_df_path):
                ref_df_codes = pd.read_csv(ref_df_path, sep='\t', encoding='gbk', dtype=str)
                stock_codes_to_fetch = ref_df_codes['代码'].str.replace('= "', '').str.replace('"', '').tolist()
                print(f"   从参考数据获取了 {len(stock_codes_to_fetch)} 个股票代码。")
        except Exception as e:
            print(f"   ❌ 无法从参考数据获取股票代码: {e}")
            print("   请确保 '参考数据/Table.xls' 存在且格式正确。")
            return

    # 限制获取数量，避免API限制或时间过长
    # stock_codes_to_fetch = stock_codes_to_fetch[:100] # 调试时可以限制数量

    for i, code in enumerate(stock_codes_to_fetch):
        if i % 500 == 0: # 每500只股票打印一次进度
            print(f"     正在获取第 {i}/{len(stock_codes_to_fetch)} 只股票的历史数据...")
        hist_df = get_stock_data(code, start_date, end_date)
        if not hist_df.empty:
            hist_df['原始代码'] = code
            all_historical_data.append(hist_df)

    if not all_historical_data:
        print("   ❌ 未能获取任何股票的历史数据，无法进行模型训练和信号生成。")
        return

    df_historical = pd.concat(all_historical_data)
    print(f"   ✅ 成功获取 {len(df_historical)} 条历史数据。")

    # 计算技术指标
    print("   计算技术指标...")
    df_historical_with_indicators = df_historical.groupby('原始代码').apply(calculate_technical_indicators)
    df_historical_with_indicators.reset_index(level=0, inplace=True) # 恢复原始代码列
    print("   ✅ 技术指标计算完成。")

    # 合并实时数据和历史数据 (以历史数据为主，补充实时数据)
    # 找到最新的历史数据日期
    latest_hist_date = df_historical_with_indicators.index.max()
    print(f"   最新历史数据日期: {latest_hist_date.strftime('%Y-%m-%d')}")

    # 提取实时数据中最新的数据，并将其日期设置为最新历史数据日期+1天 (模拟下一个交易日)
    # 或者更准确地，将实时数据作为当日数据，并确保其指标计算正确
    df_current_day_data = df_realtime.copy()
    df_current_day_data['日期'] = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) # 设置为今日日期
    df_current_day_data.set_index('日期', inplace=True)

    # 将实时数据中的列名映射到历史数据中的列名，以便计算指标
    df_current_day_data.rename(columns={
        '最新': '收盘', '涨幅%': '涨跌幅', '实际换手%': '换手率'
    }, inplace=True)

    # 确保实时数据包含所有计算指标所需的列
    for col in ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']:
        if col not in df_current_day_data.columns:
            df_current_day_data[col] = np.nan

    # 填充成交量和成交额的缺失值，避免技术指标计算失败
    df_current_day_data['成交量'] = df_current_day_data['成交量'].fillna(0)
    df_current_day_data['成交额'] = df_current_day_data['成交额'].fillna(0)

    # 将实时数据追加到历史数据中，然后重新计算所有指标
    # 这样做可以确保实时数据的技术指标是基于最新的历史数据计算的
    df_combined = pd.concat([df_historical_with_indicators, df_current_day_data.drop(columns=['代码', '名称', '所属行业', '市盈率(动)', '总市值', '归属净利润', '昨收', 'Unnamed: 16'], errors='ignore')])
    df_combined_with_indicators = df_combined.groupby('原始代码').apply(calculate_technical_indicators)
    df_combined_with_indicators.reset_index(level=0, inplace=True)

    # 提取今天的最新数据 (用于信号生成)
    df_today_features = df_combined_with_indicators[df_combined_with_indicators.index == datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)].copy()

    # 将实时数据中的名称、行业、市盈率、总市值、归属净利润等信息合并回df_today_features
    # 使用原始代码作为键进行合并
    df_today_features = df_today_features.merge(
        df_realtime[['原始代码', '名称', '所属行业', '市盈率(动)', '总市值', '归属净利润']],
        on='原始代码',
        how='left'
    )
    # 确保'最新'价是实时数据中的最新价，而不是历史收盘价
    df_today_features['最新'] = df_today_features['收盘'] # 实时数据中'最新'价被映射为'收盘'
    df_today_features['涨幅%'] = df_today_features['涨跌幅'] # 实时数据中'涨幅%'被映射为'涨跌幅'


    # 保存整合后的数据 (可选)
    # df_combined_with_indicators.to_csv('输出数据/整合历史与实时数据.csv', encoding='utf-8-sig')
    # print("   ✅ 整合后的历史与实时数据已保存。")

    # ========== 第二步：训练机器学习模型 ==========
    print("\n2. 训练机器学习模型...")

    # 训练短期模型
    short_term_model, short_term_scaler, short_term_features, short_term_indices = train_ml_model(df_combined_with_indicators.copy(), SHORT_TERM_FUTURE_DAYS)

    # 训练长期模型
    long_term_model, long_term_scaler, long_term_features, long_term_indices = train_ml_model(df_combined_with_indicators.copy(), LONG_TERM_FUTURE_DAYS)

    if short_term_model is None or long_term_model is None:
        print("   ❌ 至少一个模型训练失败，无法进行后续信号生成。")
        return

    # ========== 第三步：生成交易信号 ==========
    signals_df = generate_trading_signals(
        df_today_features,
        short_term_model, short_term_scaler, short_term_features, short_term_indices,
        long_term_model, long_term_scaler, long_term_features, long_term_indices
    )

    # 保存交易信号
    output_file_signals = '输出数据/交易信号.csv'
    signals_df.to_csv(output_file_signals, index=False, encoding='utf-8-sig')
    print(f"\n✅ 交易信号已保存: {output_file_signals}")
    print("\n🎯 今日交易信号概览 (前20名):")
    print(signals_df.head(20).to_string())

    # ========== 第四步：关联规则挖掘 ==========
    # 对所有历史数据进行关联规则挖掘，以发现普遍规律
    perform_association_rule_mining(df_historical_with_indicators.copy())

    # ========== 第五步：简化的回测 ==========
    # 注意：这里只是一个非常简化的回测示例，实际回测需要更复杂的历史数据和交易模拟
    # 这里的回测是基于当前生成的信号，模拟在今天进行交易，并假设这些交易是成功的。
    # 真正的回测需要将模型应用于历史每一天的数据，并模拟交易过程。
    simplified_backtest(signals_df.copy())

    print("\n" + "="*60)
    print("✅ 程序执行完成！")
    print("="*60)


if __name__ == "__main__":
    main()
