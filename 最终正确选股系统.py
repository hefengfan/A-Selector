#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级量化选股系统 - 实时计算版 (集成多维度数据、精密ML模型与策略回测)

核心改进点：
1.  **数据获取与整合：** 获取历史日K线数据和财务指标数据。
2.  **多维度特征工程：** 整合技术指标、基本面指标和苏氏量化策略特征。
3.  **更精密的机器学习模型：**
    *   目标变量：未来 N 日的最高收益率。
    *   模型：XGBoostRegressor。
    *   优化：Optuna 超参数优化。
    *   验证：TimeSeriesSplit 时间序列交叉验证，避免未来数据泄露。
4.  **策略生成与信号：** 提供短期/长期买入/卖出/持有信号，并建议止损止盈位。
5.  **简化的回测系统：** 模拟历史交易，评估策略表现。
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
import time # 用于API调用间隔
warnings.filterwarnings('ignore')

# 导入机器学习相关库
from sklearn.model_selection import TimeSeriesSplit
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
HISTORY_YEARS = 3 # 获取过去3年的数据
# 预测未来收益的天数 (短期/长期)
SHORT_TERM_FUTURE_DAYS = 5  # 预测未来5个交易日内的最高涨幅
LONG_TERM_FUTURE_DAYS = 20  # 预测未来20个交易日内的最高涨幅

# 信号生成阈值 (预测的未来最高收益率)
BUY_THRESHOLD_SHORT = 0.03 # 短期预测最高涨幅高于此值则买入 (3%)
BUY_THRESHOLD_LONG = 0.08  # 长期预测最高涨幅高于此值则买入 (8%)
SELL_THRESHOLD_SHORT = -0.02 # 短期预测最高跌幅低于此值则卖出 (-2%)
SELL_THRESHOLD_LONG = -0.05  # 长期预测最高跌幅低于此值则卖出 (-5%)

# 止损止盈百分比 (基于买入价)
STOP_LOSS_PERCENT = 0.05 # 5% 止损
TAKE_PROFIT_PERCENT = 0.10 # 10% 止盈

# 关联规则挖掘参数
AR_MIN_SUPPORT = 0.005 # 降低支持度以发现更多规则
AR_MIN_THRESHOLD = 1.2 # 提高提升度以发现更强的关联

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

def get_stock_hist_data(symbol, start_date, end_date):
    """
    获取指定股票的历史日K线数据 (前复权)。
    """
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        for col in ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['股票代码'] = symbol # 添加股票代码列
        return df
    except Exception as e:
        # print(f"获取 {symbol} 历史数据失败: {e}")
        return pd.DataFrame()

def get_financial_data(symbol):
    """
    获取指定股票的最新财务分析指标数据。
    """
    try:
        df = ak.stock_financial_analysis_indicator_em(symbol=symbol)
        # 选取最新一期的财务数据
        if not df.empty:
            df['报告日期'] = pd.to_datetime(df['报告日期'])
            df = df.sort_values(by='报告日期', ascending=False).iloc[0:1] # 取最新一期
            df.rename(columns={
                '市盈率(TTM)': 'PE_TTM', '市净率': 'PB', '净资产收益率': 'ROE',
                '营业总收入同比增长': 'Revenue_Growth', '归属净利润同比增长': 'NetProfit_Growth'
            }, inplace=True)
            # 确保这些列存在且为数值
            for col in ['PE_TTM', 'PB', 'ROE', 'Revenue_Growth', 'NetProfit_Growth']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df[['PE_TTM', 'PB', 'ROE', 'Revenue_Growth', 'NetProfit_Growth']].iloc[0]
        return pd.Series()
    except Exception as e:
        # print(f"获取 {symbol} 财务数据失败: {e}")
        return pd.Series()

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
    df['SMA_60'] = ta.trend.sma_indicator(df['收盘'], window=60) # 用于苏氏策略
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
    df['BB_Percent'] = bollinger.bollinger_rband() # %B指标

    # ATR (Average True Range)
    df['ATR'] = ta.volatility.average_true_range(df['最高'], df['最低'], df['收盘'], window=14)

    # 成交量移动平均
    df['Volume_SMA_5'] = ta.volume.volume_sma_indicator(df['成交量'], window=5)
    df['Volume_SMA_10'] = ta.volume.volume_sma_indicator(df['成交量'], window=10)

    # 价格变动率
    df['Daily_Return'] = df['收盘'].pct_change()
    df['Log_Return'] = np.log(df['收盘'] / df['收盘'].shift(1))

    # 苏氏策略特征 (基于计算出的均线)
    df['Price_Pos_F'] = 0
    # 确保SMA_60和SMA_20不为0或NaN
    df.loc[(df['SMA_60'] > 0) & (df['最低'] / df['SMA_60'] >= 0.85) & (df['最低'] / df['SMA_60'] <= 1.15), 'Price_Pos_F'] = 1
    df.loc[(df['SMA_20'] > 0) & (df['收盘'] / df['SMA_20'] >= 0.90) & (df['收盘'] / df['SMA_20'] <= 1.10), 'Price_Pos_F'] = 1

    df['Price_Pos_G'] = 0
    # 确保最高和最低价有效，避免除以零
    df.loc[(df['涨跌幅'] >= 5.0) & (df['最高'] - df['最低'] > 0) & (df['收盘'] >= (df['最高'] - (df['最高'] - df['最低']) * 0.30)), 'Price_Pos_G'] = 1
    df.loc[(df['涨跌幅'] >= 5.0) & (df['最高'] == df['最低']) & (df['收盘'] == df['最高']), 'Price_Pos_G'] = 1 # 涨停板情况

    # 填充计算后可能出现的NaN值，通常用0或前一个有效值填充
    df = df.fillna(method='ffill').fillna(0) # 先向前填充，再用0填充剩余的（通常是开头几行）

    return df

def prepare_data_for_ml(df_all_stocks, future_days):
    """
    准备用于机器学习的数据集，包括特征和目标变量。
    目标变量为未来 N 日内的最高收益率。
    """
    features_list = []
    targets_list = []
    stock_codes_list = []
    dates_list = []

    # 定义所有可能用到的特征列
    feature_cols = [
        '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率',
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_60', 'EMA_5', 'EMA_10', 'EMA_20',
        'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'K', 'D', 'J',
        'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Percent', 'ATR',
        'Volume_SMA_5', 'Volume_SMA_10', 'Daily_Return', 'Log_Return',
        'Price_Pos_F', 'Price_Pos_G',
        'PE_TTM', 'PB', 'ROE', 'Revenue_Growth', 'NetProfit_Growth' # 财务指标
    ]

    # 遍历每只股票，计算其特征和未来收益
    for stock_code, stock_df in df_all_stocks.groupby('股票代码'):
        stock_df = stock_df.sort_index() # 确保按日期排序

        # 计算未来 N 日内的最高收益率作为目标变量
        # 避免未来数据泄露：使用 shift(-future_days) 获取未来的收盘价
        # 然后计算未来 N 日内的最高价与当前收盘价的涨幅
        stock_df['future_max_price'] = stock_df['最高'].rolling(window=future_days, closed='right').max().shift(-future_days + 1)
        stock_df['future_max_return'] = (stock_df['future_max_price'] / stock_df['收盘']) - 1

        # 提取特征和目标
        for i in range(len(stock_df)):
            row = stock_df.iloc[i]
            # 确保目标变量存在且不是NaN
            if pd.notna(row['future_max_return']):
                current_features = []
                for col in feature_cols:
                    current_features.append(row.get(col, np.nan)) # 使用.get()避免KeyError

                # 过滤掉特征中包含NaN或Inf的行
                if not any(pd.isna(f) or np.isinf(f) for f in current_features):
                    features_list.append(current_features)
                    targets_list.append(row['future_max_return'])
                    stock_codes_list.append(stock_code)
                    dates_list.append(row.name) # 记录日期

    X = np.array(features_list)
    y = np.array(targets_list)

    # 再次检查并移除包含 NaN 或无穷大的行 (双重保险)
    mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1) & ~np.isnan(y) & ~np.isinf(y)
    X = X[mask]
    y = y[mask]
    stock_codes_filtered = np.array(stock_codes_list)[mask]
    dates_filtered = np.array(dates_list)[mask]

    return X, y, stock_codes_filtered, dates_filtered, feature_cols

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
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    }

    if param['booster'] == 'gbtree':
        param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
        param['max_depth'] = trial.suggest_int('max_depth', 3, 9)
    elif param['booster'] == 'dart':
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
    训练机器学习模型，预测股票未来最高收益率。
    使用 TimeSeriesSplit 进行交叉验证。
    """
    print(f"\n   准备训练数据 (预测未来 {future_days} 天最高收益率)...")
    X, y, stock_codes, dates, feature_names = prepare_data_for_ml(df_all_stocks.copy(), future_days)

    if len(X) < 100: # 至少需要足够的数据来训练和验证
        print(f"   ❌ 有效训练数据不足 ({len(X)} 条)，无法训练模型。")
        return None, None, None

    print(f"   有效训练样本数: {len(X)}")

    # 数据预处理
    print("   数据预处理 (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # TimeSeriesSplit 划分训练集和测试集
    # n_splits 决定了有多少个训练/测试对，max_train_size 限制了训练集的大小
    # test_size 决定了每个测试集的大小
    tscv = TimeSeriesSplit(n_splits=5) # 5折时间序列交叉验证

    best_model = None
    best_rmse = float('inf')
    
    print("   启动 Optuna 超参数优化 (可能需要一些时间)...")
    # Optuna 优化在每次交叉验证的第一个折叠上进行，以找到最佳参数
    # 实际应用中，可以先用一部分数据找到最佳参数，再用全部数据训练
    
    # 仅在第一个折叠上进行 Optuna 优化，以节省时间
    for fold, (train_index, test_index) in enumerate(tscv.split(X_scaled)):
        if fold == 0: # 只在第一个折叠进行超参数搜索
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]

            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
            try:
                study.optimize(lambda trial: objective_xgb(trial, X_train, y_train, X_test, y_test), n_trials=30, show_progress_bar=True) # 减少试用次数以加快速度
            except Exception as e:
                print(f"   Optuna 优化过程中发生错误: {e}")
                print("   将使用默认或预设参数训练模型。")
                best_params = {
                    'objective': 'reg:squareerror', 'eval_metric': 'rmse', 'booster': 'gbtree',
                    'lambda': 1, 'alpha': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                    'min_child_weight': 1, 'learning_rate': 0.05, 'n_estimators': 500,
                    'gamma': 0, 'max_depth': 6, 'seed': 42, 'n_jobs': -1,
                }
            else:
                print("\n   Optuna 优化完成。")
                print(f"   最佳均方误差 (RMSE): {study.best_value:.4f}")
                print(f"   最佳超参数: {study.best_params}")
                best_params = study.best_params
            break # 找到最佳参数后退出循环

    # 使用最佳参数在所有数据上训练最终模型
    print("   使用最佳参数训练最终 XGBoost 模型...")
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_scaled, y) # 使用全部数据进行最终训练

    # 评估模型 (在整个数据集上进行一次最终评估)
    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"   最终模型均方误差 (MSE): {mse:.4f}")
    print(f"   最终模型R²分数: {r2:.4f}")

    return model, scaler, feature_names

def predict_future_return(row, model, scaler, feature_names):
    """
    使用训练好的模型预测股票未来最高收益率。
    """
    current_features = []
    for col in feature_names:
        current_features.append(safe_float(row.get(col, np.nan))) # 使用.get()避免KeyError

    if any(pd.isna(f) or np.isinf(f) for f in current_features):
        return np.nan

    features_array = np.array(current_features).reshape(1, -1)
    try:
        features_scaled = scaler.transform(features_array)
        predicted_return = model.predict(features_scaled)[0]
        return predicted_return
    except Exception as e:
        # print(f"预测未来收益时发生错误: {e}, 特征: {current_features}")
        return np.nan

def generate_trading_signals(df_current_data, short_term_model, short_term_scaler, short_term_features,
                             long_term_model, long_term_scaler, long_term_features):
    """
    根据模型预测生成买入/卖出信号，并建议止损止盈位。
    """
    print("\n3. 生成交易信号...")
    signals = []

    # 确保所有数值列都是float类型
    for col in ['最新', '涨幅%', '最高', '最低', '实际换手%', '市盈率(动)', '总市值', '归属净利润', '昨收', '开盘']:
        if col in df_current_data.columns:
            df_current_data[col] = df_current_data[col].apply(safe_float)

    for idx, row in df_current_data.iterrows():
        stock_code = str(row['股票代码']).strip()
        stock_name = str(row['名称']).strip()
        industry = str(row['所属行业']).strip()
        current_price = safe_float(row['最新'])
        current_change = safe_float(row['涨幅%'])

        short_pred_return = np.nan
        long_pred_return = np.nan
        signal_short = "持有"
        signal_long = "持有"
        overall_signal = "持有"
        stop_loss_price = np.nan
        take_profit_price = np.nan

        # 预测短期收益
        if short_term_model and short_term_scaler and short_term_features:
            short_pred_return = predict_future_return(row, short_term_model, short_term_scaler, short_term_features)
            if pd.notna(short_pred_return):
                if short_pred_return >= BUY_THRESHOLD_SHORT:
                    signal_short = "短期买入"
                elif short_pred_return < SELL_THRESHOLD_SHORT:
                    signal_short = "短期卖出"

        # 预测长期收益
        if long_term_model and long_term_scaler and long_term_features:
            long_pred_return = predict_future_return(row, long_term_model, long_term_scaler, long_term_features)
            if pd.notna(long_pred_return):
                if long_pred_return >= BUY_THRESHOLD_LONG:
                    signal_long = "长期买入"
                elif long_pred_return < SELL_THRESHOLD_LONG:
                    signal_long = "长期卖出"

        # 综合信号判断
        if signal_short == "短期买入" and signal_long == "长期买入":
            overall_signal = "强烈买入"
        elif signal_short == "短期买入" and signal_long == "持有":
            overall_signal = "短期买入"
        elif signal_long == "长期买入" and signal_short == "持有":
            overall_signal = "长期买入"
        elif signal_short == "短期卖出" or signal_long == "长期卖出":
            overall_signal = "卖出"

        # 计算止损止盈价 (仅对买入信号建议)
        if overall_signal in ["强烈买入", "短期买入", "长期买入"] and pd.notna(current_price) and current_price > 0:
            stop_loss_price = current_price * (1 - STOP_LOSS_PERCENT)
            take_profit_price = current_price * (1 + TAKE_PROFIT_PERCENT)

        signals.append({
            '代码': stock_code,
            '名称': stock_name,
            '行业': industry,
            '最新价': f"{current_price:.2f}" if pd.notna(current_price) else "--",
            '今日涨幅': f"{current_change:.2f}%" if pd.notna(current_change) else "--",
            f'预测{SHORT_TERM_FUTURE_DAYS}日最高收益': f"{short_pred_return*100:.2f}%" if pd.notna(short_pred_return) else "--",
            f'预测{LONG_TERM_FUTURE_DAYS}日最高收益': f"{long_pred_return*100:.2f}%" if pd.notna(long_pred_return) else "--",
            '短期信号': signal_short,
            '长期信号': signal_long,
            '综合信号': overall_signal,
            '建议止损价': f"{stop_loss_price:.2f}" if pd.notna(stop_loss_price) else "--",
            '建议止盈价': f"{take_profit_price:.2f}" if pd.notna(take_profit_price) else "--"
        })
    
    signals_df = pd.DataFrame(signals)
    # 排序，优先显示强烈买入，然后是长期买入，短期买入，再按预测收益排序
    signal_order = {"强烈买入": 4, "长期买入": 3, "短期买入": 2, "持有": 1, "卖出": 0}
    signals_df['signal_rank'] = signals_df['综合信号'].map(signal_order)
    
    # 将预测收益转换为数值以便排序
    signals_df[f'预测{SHORT_TERM_FUTURE_DAYS}日最高收益_num'] = signals_df[f'预测{SHORT_TERM_FUTURE_DAYS}日最高收益'].str.replace('%', '').apply(safe_float) / 100
    signals_df[f'预测{LONG_TERM_FUTURE_DAYS}日最高收益_num'] = signals_df[f'预测{LONG_TERM_FUTURE_DAYS}日最高收益'].str.replace('%', '').apply(safe_float) / 100

    signals_df = signals_df.sort_values(by=['signal_rank', f'预测{LONG_TERM_FUTURE_DAYS}日最高收益_num', f'预测{SHORT_TERM_FUTURE_DAYS}日最高收益_num'], ascending=[False, False, False])
    
    # 删除辅助列
    signals_df = signals_df.drop(columns=['signal_rank', f'预测{SHORT_TERM_FUTURE_DAYS}日最高收益_num', f'预测{LONG_TERM_FUTURE_DAYS}日最高收益_num'])

    return signals_df

def perform_association_rule_mining(df):
    """
    使用关联规则挖掘来发现策略条件与高涨幅之间的关系。
    """
    print("\n4. 执行关联规则挖掘...")

    data_for_ar = []
    # 确保所有数值列都是float类型
    numeric_cols = [
        '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率',
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_60', 'EMA_5', 'EMA_10', 'EMA_20',
        'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'K', 'D', 'J',
        'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Percent', 'ATR',
        'Volume_SMA_5', 'Volume_SMA_10', 'Daily_Return', 'Log_Return',
        'Price_Pos_F', 'Price_Pos_G',
        'PE_TTM', 'PB', 'ROE', 'Revenue_Growth', 'NetProfit_Growth'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)

    for _, row in df.iterrows():
        items = []

        # 苏氏策略特征
        f_condition = safe_float(row.get('Price_Pos_F'))
        g_condition = safe_float(row.get('Price_Pos_G'))
        profit_growth = safe_float(row.get('NetProfit_Growth'))
        turnover = safe_float(row.get('换手率'))
        pe_ratio = safe_float(row.get('PE_TTM'))
        rsi = safe_float(row.get('RSI'))

        if pd.notna(f_condition):
            items.append("F_价格位置_满足" if f_condition == 1 else "F_价格位置_不满足")
        if pd.notna(g_condition):
            items.append("G_涨幅位置_满足" if g_condition == 1 else "G_涨幅位置_不满足")
        if pd.notna(profit_growth):
            items.append("净利润增长_高" if profit_growth >= 20 else "净利润增长_低") # 20%增长
        if pd.notna(turnover):
            items.append("换手率_适中" if 1 <= turnover <= 10 else "换手率_极端") # 1%-10%为适中
        if pd.notna(pe_ratio):
            items.append("市盈率_低" if pe_ratio > 0 and pe_ratio <= 30 else "市盈率_高")
        if pd.notna(rsi):
            items.append("RSI_超买" if rsi >= 70 else ("RSI_超卖" if rsi <= 30 else "RSI_中性"))

        # 目标变量：高涨幅 (例如，当日涨幅 > 5%)
        change = safe_float(row.get('涨跌幅'))
        if pd.notna(change) and change > 5.0: # 可以调整这个阈值
            items.append("当日高涨幅")
        else:
            items.append("当日低涨幅")

        if items: # 确保有有效项
            data_for_ar.append(items)

    if not data_for_ar:
        print("   ❌ 没有足够的数据进行关联规则挖掘。")
        return

    te = TransactionEncoder()
    te_ary = te.fit(data_for_ar).transform(data_for_ar)
    df_ar = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_ar, min_support=AR_MIN_SUPPORT, use_colnames=True)
    if frequent_itemsets.empty:
        print("   ⚠️ 未找到频繁项集，请尝试降低 min_support。")
        return

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=AR_MIN_THRESHOLD)

    if rules.empty:
        print("   ⚠️ 未找到有意义的关联规则，请尝试降低 min_threshold 或检查数据。")
        return

    high_return_rules = rules[rules['consequents'].apply(lambda x: '当日高涨幅' in x)]
    high_return_rules = high_return_rules.sort_values(by=['lift', 'confidence'], ascending=False)

    print("\n   发现以下与 '当日高涨幅' 相关的关联规则 (按 Lift 降序):")
    if high_return_rules.empty:
        print("   未找到直接导致 '当日高涨幅' 的关联规则。")
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
    注意：这是一个非常简化的版本，不考虑滑点、佣金、资金管理、盘中价格波动等复杂因素。
    仅用于初步评估策略的潜在方向。
    """
    print("\n5. 简化的策略回测 (仅供参考，不含真实交易细节)...")
    
    # 假设我们只关注“强烈买入”和“卖出”信号
    buy_signals = signals_df[signals_df['综合信号'] == '强烈买入']
    sell_signals = signals_df[signals_df['综合信号'] == '卖出']

    print(f"   初始资金: {initial_capital:.2f} 元")
    current_capital = initial_capital
    positions = {} # {股票代码: {'quantity': 数量, 'avg_price': 均价}}
    trade_log = []

    # 模拟交易 (非常粗略的逻辑)
    # 假设我们只在信号生成当天进行交易，并且只买入强烈买入的股票，卖出卖出信号的股票
    # 并且假设我们能以最新价成交

    # 卖出操作 (先处理卖出，释放资金)
    for _, row in sell_signals.iterrows():
        code = row['代码']
        if code in positions and positions[code]['quantity'] > 0:
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
    if not trade_log:
        print("   无交易发生。")
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
    print("高级量化选股系统 - 实时计算版 (集成机器学习与多策略)")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 创建输出目录
    os.makedirs('输出数据', exist_ok=True)

    # ========== 第一步：获取并整合数据 ==========
    print("\n1. 获取A股数据 (实时 + 历史K线 + 财务指标)...")

    # 1.1 获取所有A股实时数据 (用于获取股票列表和今日实时行情)
    df_realtime = pd.DataFrame()
    try:
        print("   尝试获取实时数据...")
        df_realtime = ak.stock_zh_a_spot_em()
        print(f"   ✅ 成功获取 {len(df_realtime)} 只股票的实时数据")

        df_realtime.rename(columns={
            '最新价': '最新', '涨跌幅': '涨幅%', '换手率': '实际换手%',
            '市盈率-动态': '市盈率(动)', '总市值': '总市值', '归属净利润': '归属净利润'
        }, inplace=True)
        df_realtime['股票代码'] = df_realtime['代码'].copy() # 统一列名
        df_realtime['代码'] = df_realtime['代码'].apply(lambda x: f'= "{str(x)}"') # 格式化代码用于Excel

        # 确保所有关键列存在
        required_cols_realtime = ['代码', '名称', '最新', '涨幅%', '最高', '最低', '实际换手%',
                                  '所属行业', '市盈率(动)', '总市值', '归属净利润', '昨收', '开盘', '股票代码']
        for col in required_cols_realtime:
            if col not in df_realtime.columns:
                df_realtime[col] = np.nan

    except Exception as e:
        print(f"   ❌ 实时数据获取失败: {e}")
        print("   将尝试从参考数据获取股票代码。")
        df_realtime = pd.DataFrame(columns=required_cols_realtime) # 创建空DataFrame以避免后续错误

    stock_codes_to_fetch = df_realtime['股票代码'].tolist()
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

    # 1.2 获取历史数据并计算指标
    all_historical_data = []
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=HISTORY_YEARS * 365)).strftime('%Y%m%d') # 3年历史数据

    print(f"   获取历史日K线数据 ({start_date} 至 {end_date})...")
    # 限制获取数量，避免API限制或时间过长 (调试时可以取消注释)
    # stock_codes_to_fetch = stock_codes_to_fetch[:50] 

    for i, code in enumerate(stock_codes_to_fetch):
        if i % 100 == 0: # 每100只股票打印一次进度
            print(f"     正在获取第 {i}/{len(stock_codes_to_fetch)} 只股票的历史数据...")
        hist_df = get_stock_hist_data(code, start_date, end_date)
        if not hist_df.empty:
            all_historical_data.append(hist_df)
        time.sleep(0.05) # 避免API频率限制

    if not all_historical_data:
        print("   ❌ 未能获取任何股票的历史数据，无法进行模型训练和信号生成。")
        return

    df_historical = pd.concat(all_historical_data)
    print(f"   ✅ 成功获取 {len(df_historical)} 条历史K线数据。")

    # 1.3 获取财务数据并合并
    print("   获取财务指标数据 (最新一期)...")
    all_financial_data = []
    for i, code in enumerate(stock_codes_to_fetch):
        if i % 100 == 0:
            print(f"     正在获取第 {i}/{len(stock_codes_to_fetch)} 只股票的财务数据...")
        fin_data = get_financial_data(code)
        if not fin_data.empty:
            fin_data['股票代码'] = code
            all_financial_data.append(fin_data)
        time.sleep(0.05) # 避免API频率限制

    if all_financial_data:
        df_financial = pd.DataFrame(all_financial_data).set_index('股票代码')
        # 将财务数据合并到历史数据中 (按股票代码合并，并向前填充，假设财务数据在报告期后一直有效)
        df_historical = df_historical.reset_index().set_index('股票代码')
        df_historical = df_historical.merge(df_financial, left_index=True, right_index=True, how='left')
        df_historical = df_historical.reset_index().set_index('日期') # 恢复日期索引
        print(f"   ✅ 成功获取并合并 {len(df_financial)} 只股票的财务数据。")
    else:
        print("   ⚠️ 未能获取任何股票的财务数据。")

    # 1.4 计算技术指标
    print("   计算技术指标...")
    # 对每个股票分组计算技术指标，确保指标计算的正确性
    df_historical_with_indicators = df_historical.groupby('股票代码', group_keys=False).apply(calculate_technical_indicators)
    print("   ✅ 技术指标计算完成。")

    # 1.5 准备今日数据用于预测
    # 将实时数据中的列名映射到历史数据中的列名，以便计算指标
    df_current_day_data = df_realtime.copy()
    df_current_day_data['日期'] = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) # 设置为今日日期
    df_current_day_data.set_index('日期', inplace=True)
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
    # 注意：这里只追加K线相关数据，财务数据已经合并到df_historical_with_indicators中
    df_combined_for_indicators = pd.concat([
        df_historical_with_indicators[['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率', '股票代码']],
        df_current_day_data[['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率', '股票代码']]
    ])
    df_combined_for_indicators = df_combined_for_indicators.groupby('股票代码', group_keys=False).apply(calculate_technical_indicators)

    # 提取今天的最新数据 (用于信号生成)
    df_today_features = df_combined_for_indicators[df_combined_for_indicators.index == datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)].copy()
    
    # 将实时数据中的名称、行业、市盈率、总市值、归属净利润等信息合并回df_today_features
    df_today_features = df_today_features.merge(
        df_realtime[['股票代码', '名称', '所属行业', '市盈率(动)', '总市值', '归属净利润']],
        on='股票代码',
        how='left'
    )
    # 确保'最新'价是实时数据中的最新价，而不是历史收盘价
    df_today_features['最新'] = df_today_features['收盘'] # 实时数据中'最新'价被映射为'收盘'
    df_today_features['涨幅%'] = df_today_features['涨跌幅'] # 实时数据中'涨幅%'被映射为'涨跌幅'

    # 将财务数据合并到df_today_features
    if 'df_financial' in locals() and not df_financial.empty:
        df_today_features = df_today_features.merge(df_financial.reset_index(), on='股票代码', how='left')
    
    # 最终用于模型训练的数据集 (包含所有历史K线、技术指标和财务数据)
    df_for_training = df_historical_with_indicators.copy()
    if 'df_financial' in locals() and not df_financial.empty:
        df_for_training = df_for_training.reset_index().set_index('股票代码')
        df_for_training = df_for_training.merge(df_financial, left_index=True, right_index=True, how='left')
        df_for_training = df_for_training.reset_index().set_index('日期')


    # 保存整合后的数据 (可选)
    # df_for_training.to_csv('输出数据/整合历史与财务数据_训练集.csv', encoding='utf-8-sig')
    # df_today_features.to_csv('输出数据/今日预测特征集.csv', encoding='utf-8-sig')
    # print("   ✅ 整合后的数据已保存。")

    # ========== 第二步：训练机器学习模型 ==========
    print("\n2. 训练机器学习模型...")
    
    # 训练短期模型
    short_term_model, short_term_scaler, short_term_features = train_ml_model(df_for_training.copy(), SHORT_TERM_FUTURE_DAYS)
    
    # 训练长期模型
    long_term_model, long_term_scaler, long_term_features = train_ml_model(df_for_training.copy(), LONG_TERM_FUTURE_DAYS)

    if short_term_model is None or long_term_model is None:
        print("   ❌ 至少一个模型训练失败，无法进行后续信号生成。")
        return

    # ========== 第三步：生成交易信号 ==========
    signals_df = generate_trading_signals(
        df_today_features,
        short_term_model, short_term_scaler, short_term_features,
        long_term_model, long_term_scaler, long_term_features
    )

    # 保存交易信号
    output_file_signals = '输出数据/交易信号.csv'
    signals_df.to_csv(output_file_signals, index=False, encoding='utf-8-sig')
    print(f"\n✅ 交易信号已保存: {output_file_signals}")
    print("\n🎯 今日交易信号概览 (前20名):")
    print(signals_df.head(20).to_string())

    # ========== 第四步：关联规则挖掘 ==========
    # 对所有历史数据进行关联规则挖掘，以发现普遍规律
    perform_association_rule_mining(df_for_training.copy())

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
