#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态选股系统 - 实时计算版 (集成神经网络与关联规则)
结合专门技术优化预测模型 结合特征和预测结果给我策略 针对推荐股票 短期/长期 买入/卖出 结合专业分析进行补充 完整代码
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# 导入 Optuna 进行超参数优化
import optuna

# 导入 mlxtend 进行关联规则挖掘
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 技术分析库
import talib

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
            return float(s.replace('亿', '')) * 10000
        if '万亿' in s:
            return float(s.replace('万亿', '')) * 100000000
        if '万' in s:
            return float(s.replace('万', '')) / 10000
        return float(s)
    except ValueError:
        return np.nan

def calculate_features(row):
    """
    根据苏氏量化策略计算特征值，用于神经网络训练和关联规则挖掘。
    返回一个包含数值特征的列表。
    """
    features = []

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

    # H列：归属净利润 (数值，单位亿)
    try:
        profit = safe_float(row.get('归属净利润'))
        features.append(profit if pd.notna(profit) else 0)
    except:
        features.append(0)

    # I列：实际换手率 (数值)
    try:
        turnover = safe_float(row.get('实际换手%'))
        features.append(turnover if pd.notna(turnover) else 100) # 缺失时给一个较大值
    except:
        features.append(100)

    # J列：总市值 (数值，单位亿)
    try:
        cap = safe_float(row.get('总市值'))
        features.append(cap if pd.notna(cap) else 0)
    except:
        features.append(0)

    return features

def calculate_technical_indicators(df, code):
    """
    计算技术指标：MACD, RSI, BOLL
    """
    try:
        # 获取历史数据
        stock_df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq").iloc[-60:]  # 最近60个交易日
        
        if len(stock_df) < 20:
            return 0, 0, 0, 0
        
        # 计算MACD
        stock_df['MACD'], stock_df['MACDsignal'], stock_df['MACDhist'] = talib.MACD(
            stock_df['收盘'], fastperiod=12, slowperiod=26, signalperiod=9)
        
        # 计算RSI
        stock_df['RSI'] = talib.RSI(stock_df['收盘'], timeperiod=14)
        
        # 计算布林带
        stock_df['upper'], stock_df['middle'], stock_df['lower'] = talib.BBANDS(
            stock_df['收盘'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # 获取最新值
        last_row = stock_df.iloc[-1]
        
        # MACD信号 (1:金叉, -1:死叉, 0:无)
        macd_signal = 0
        if last_row['MACD'] > last_row['MACDsignal'] and stock_df.iloc[-2]['MACD'] <= stock_df.iloc[-2]['MACDsignal']:
            macd_signal = 1
        elif last_row['MACD'] < last_row['MACDsignal'] and stock_df.iloc[-2]['MACD'] >= stock_df.iloc[-2]['MACDsignal']:
            macd_signal = -1
        
        # RSI值
        rsi = last_row['RSI']
        
        # 布林带位置 (1:上轨, -1:下轨, 0:中轨)
        boll_position = 0
        close_price = last_row['收盘']
        if close_price > last_row['upper']:
            boll_position = 1
        elif close_price < last_row['lower']:
            boll_position = -1
        
        # 成交量变化率 (5日平均成交量/20日平均成交量)
        vol_5 = stock_df['成交量'].tail(5).mean()
        vol_20 = stock_df['成交量'].tail(20).mean()
        vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1
        
        return macd_signal, rsi, boll_position, vol_ratio
    
    except Exception as e:
        print(f"计算技术指标时出错({code}): {e}")
        return 0, 50, 0, 1

def objective(trial, X_train, y_train, X_test, y_test, target_type):
    """
    Optuna 优化目标函数
    """
    hidden_layer_sizes = []
    n_layers = trial.suggest_int('n_layers', 1, 3)
    for i in range(n_layers):
        hidden_layer_sizes.append(trial.suggest_int(f'n_units_l{i}', 16, 128))

    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
    solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
    learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-2)
    
    # 根据目标类型调整参数范围
    if target_type == 'short_term':
        learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-3, 0.1)
        n_iter_no_change = 10
    else:  # long_term
        n_iter_no_change = 30

    model = MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        random_state=42,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=n_iter_no_change,
        tol=1e-4
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def train_neural_network(df, target_type='comprehensive'):
    """
    训练神经网络模型，预测股票评分，使用 Optuna 进行超参数优化。
    支持三种目标类型：短期、长期、综合
    """
    print(f"\n   准备训练数据 ({target_type}模型)...")
    X = []
    
    for _, row in df.iterrows():
        features = calculate_features(row)
        X.append(features)

    X = np.array(X)

    # 提取用于计算质量评分的列
    change = df['涨幅%'].apply(safe_float)
    profit = df['归属净利润'].apply(safe_float)
    turnover = df['实际换手%'].apply(safe_float)
    market_cap = df['总市值'].apply(safe_float)
    pe_ratio = df['市盈率(动)'].apply(safe_float)

    # 归一化各个指标 (使用 min-max 归一化)
    scaler = MinMaxScaler()
    change_norm = scaler.fit_transform(change.values.reshape(-1, 1)).flatten()
    profit_norm = scaler.fit_transform(profit.values.reshape(-1, 1)).flatten()
    turnover_norm = scaler.fit_transform(turnover.values.reshape(-1, 1)).flatten()
    market_cap_norm = scaler.fit_transform(market_cap.values.reshape(-1, 1)).flatten()
    pe_ratio_norm = scaler.fit_transform(pe_ratio.values.reshape(-1, 1)).flatten()

    # 处理 NaN 值，用 0 填充
    change_norm = np.nan_to_num(change_norm, nan=0)
    profit_norm = np.nan_to_num(profit_norm, nan=0)
    turnover_norm = np.nan_to_num(turnover_norm, nan=0)
    market_cap_norm = np.nan_to_num(market_cap_norm, nan=0)
    pe_ratio_norm = np.nan_to_num(pe_ratio_norm, nan=0)

    # 根据目标类型计算不同的评分
    if target_type == 'short_term':
        # 短期评分: 主要关注技术面和市场情绪
        y = (0.5 * change_norm + 
             0.3 * turnover_norm + 
             0.2 * (1 - abs(turnover_norm - 0.5)))  # 换手率适中最好
        
    elif target_type == 'long_term':
        # 长期评分: 主要关注基本面和价值
        y = (0.4 * profit_norm + 
             0.3 * market_cap_norm + 
             0.2 * (1 - pe_ratio_norm) + 
             0.1 * change_norm)  # 市盈率越低越好
        
    else:  # comprehensive
        # 综合评分
        y = (0.4 * change_norm + 
             0.2 * profit_norm + 
             0.15 * (1 - abs(turnover_norm - 0.5)) + 
             0.15 * market_cap_norm + 
             0.1 * (1 - pe_ratio_norm))

    # 移除包含 NaN 或无穷大的行
    mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1) & ~np.isnan(y) & ~np.isinf(y)
    X = X[mask]
    y = y[mask]

    if len(X) < 20: # 至少需要一些数据来划分训练集和测试集
        print(f"   ❌ 有效训练数据不足，无法训练{target_type}神经网络。")
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
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, target_type), 
                       n_trials=50, show_progress_bar=True)
    except Exception as e:
        print(f"   Optuna 优化过程中发生错误: {e}")
        print("   将使用默认或预设参数训练模型。")
        # 如果Optuna失败，使用一个合理的默认配置
        best_params = {
            'n_layers': 2,
            'n_units_l0': 64,
            'n_units_l1': 32,
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
        max_iter=1000, # 增加最大迭代次数
        early_stopping=True,
        n_iter_no_change=30, # 增加耐心
        tol=1e-4 # 增加容忍度
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

def predict_score_with_nn(row, model, scaler):
    """
    使用训练好的神经网络模型预测股票评分
    """
    features = calculate_features(row)
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

def perform_association_rule_mining(df):
    """
    使用关联规则挖掘来发现苏氏量化策略条件与高涨幅之间的关系。
    """
    print("\n4. 执行关联规则挖掘...")

    # 准备数据：将特征和目标变量二值化
    data_for_ar = []
    for _, row in df.iterrows():
        features = calculate_features(row)
        items = []

        # F列：价格位置条件
        if features[0] == 1:
            items.append("F_价格位置_满足")
        else:
            items.append("F_价格位置_不满足")

        # G列：涨幅和价格位置
        if features[1] == 1:
            items.append("G_涨幅位置_满足")
        else:
            items.append("G_涨幅位置_不满足")

        # H列：净利润>=3000万 (0.3亿)
        if features[2] >= 0.3:
            items.append("H_净利润_高")
        else:
            items.append("H_净利润_低")

        # I列：换手率<=20%
        if features[3] <= 20:
            items.append("I_换手率_低")
        else:
            items.append("I_换手率_高")

        # J列：市值>=300亿
        if features[4] >= 300:
            items.append("J_市值_大")
        else:
            items.append("J_市值_小")

        # 目标变量：高涨幅 (例如，涨幅 > 2%)
        change = safe_float(row.get('涨幅%'))
        if pd.notna(change) and change > 1.0:  # 降低高涨幅阈值
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
    frequent_itemsets = apriori(df_ar, min_support=0.005, use_colnames=True) # 调整min_support
    if frequent_itemsets.empty:
        print("   ⚠️ 未找到频繁项集，请尝试降低 min_support。")
        return

    # 生成关联规则
    # min_confidence 越高，规则越可靠
    # lift > 1 表示前件和后件正相关
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0) # 调整min_threshold

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
    for col in ['最新', '涨幅%', '最高', '最低', '实际换手%', '20日均价', '60日均价', '市盈率(动)', '总市值', '归属净利润', '昨收', '开盘']:
        df[col] = df[col].apply(lambda x: f" {x:.2f}" if pd.notna(x) else " --")

    # 格式化代码和名称
    df['代码'] = df['代码'].apply(lambda x: f'= "{str(x)}"' if not str(x).startswith('=') else x)
    df['名称'] = df['名称'].apply(lambda x: f" {x}" if not str(x).startswith(' ') else x)

    final_df_for_output = df[output_columns].copy()

    # 保存A股数据
    output_file1 = '输出数据/A股数据.csv'
    final_df_for_output.to_csv(output_file1, index=False, encoding='utf-8-sig')
    print(f"\n✅ A股数据已保存: {output_file1}")
    print(f"   共 {len(final_df_for_output)} 只股票")

    # ========== 第二步：训练神经网络 ==========
    print("\n2. 训练神经网络模型...")
    print("   - 训练综合模型...")
    model_comprehensive, scaler_comprehensive = train_neural_network(df.copy(), 'comprehensive')
    
    print("\n   - 训练短期模型...")
    model_short_term, scaler_short_term = train_neural_network(df.copy(), 'short_term')
    
    print("\n   - 训练长期模型...")
    model_long_term, scaler_long_term = train_neural_network(df.copy(), 'long_term')

    if model_comprehensive is None or model_short_term is None or model_long_term is None:
        print("   ❌ 神经网络训练失败，无法进行后续筛选。")
        return

    # ========== 第三步：动态筛选优质股票 ==========
    print("\n3. 动态筛选优质股票 (基于神经网络评分)...")

    quality_stocks = []

    # 重新加载原始数值的df，因为上面为了输出csv已经格式化了
    df_for_scoring = df.copy()
    for col in ['最新', '涨幅%', '最高', '最低', '实际换手%', '20日均价', '60日均价', '市盈率(动)', '总市值', '归属净利润', '昨收', '开盘']:
        df_for_scoring[col] = df_for_scoring[col].apply(safe_float)
    df_for_scoring['原始代码'] = df_for_scoring['代码'].apply(lambda x: str(x).replace('= "', '').replace('"', ''))


    for idx, row in df_for_scoring.iterrows():
        code = str(row['原始代码']).strip()
        # 计算综合评分
        comprehensive_score = predict_score_with_nn(row, model_comprehensive, scaler_comprehensive)
        
        # 计算短期评分
        short_term_score = predict_score_with_nn(row, model_short_term, scaler_short_term)
        
        # 计算长期评分
        long_term_score = predict_score_with_nn(row, model_long_term, scaler_long_term)
        
        # 计算技术指标
        macd, rsi, boll, vol_ratio = calculate_technical_indicators(df_for_scoring, code)

        if pd.notna(comprehensive_score) and pd.notna(short_term_score) and pd.notna(long_term_score):
            quality_stocks.append({
                '代码': code,
                '名称': str(row['名称']).strip(),
                '行业': str(row['所属行业']).strip(),
                '综合评分': comprehensive_score,
                '短期评分': short_term_score,
                '长期评分': long_term_score,
                '涨幅': safe_float(row['涨幅%']),
                '总市值': safe_float(row['总市值']),
                '换手率': safe_float(row['实际换手%']),
                '市盈率(动)': safe_float(row['市盈率(动)']),
                'MACD': macd,
                'RSI': rsi,
                'BOLL': boll,
                '成交量比': vol_ratio
            })

    # 按综合评分降序排序
    quality_stocks = sorted(quality_stocks, key=lambda x: (x['综合评分'], x['代码']), reverse=True)

    # 确定筛选阈值：取前N个，或者根据分数分布动态调整
    display_count = 15 # 默认显示前15个
    if len(quality_stocks) > display_count:
        # 如果股票数量足够，取前N个的最低分数作为阈值
        threshold = quality_stocks[display_count-1]['综合评分']
        quality_stocks_filtered = quality_stocks[:display_count]
    elif len(quality_stocks) > 0:
        threshold = quality_stocks[-1]['综合评分'] # 所有股票的最低分
        quality_stocks_filtered = quality_stocks
    else:
        threshold = 0.0
        quality_stocks_filtered = []

    # 保存优质股票
    output_file2 = '输出数据/优质股票.txt'
    with open(output_file2, 'w', encoding='utf-8') as f:
        f.write("苏氏量化策略 - 优质股票筛选结果 (神经网络评分)\n")
        f.write(f"筛选时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"最低综合评分阈值 (基于前{display_count}名或全部): {threshold:.4f}\n")
        f.write(f"优质股票数量: {len(quality_stocks_filtered)}\n")
        f.write("="*50 + "\n\n")

        for stock in quality_stocks_filtered:
            f.write(f"股票代码: {stock['代码']}\n")
            f.write(f"股票名称: {stock['名称']}\n")
            f.write(f"所属行业: {stock['行业']}\n")
            f.write(f"综合评分: {stock['综合评分']:.4f}\n")
            f.write(f"短期评分: {stock['短期评分']:.4f}\n")
            f.write(f"长期评分: {stock['长期评分']:.4f}\n")
            f.write(f"今日涨幅: {stock['涨幅']:.2f}%\n")
            f.write(f"总市值: {stock['总市值']:.2f} 亿\n")
            f.write(f"换手率: {stock['换手率']:.2f}%\n")
            f.write(f"市盈率(动): {stock['市盈率(动)']:.2f}\n")
            f.write(f"技术指标 - MACD信号: {stock['MACD']}, RSI: {stock['RSI']:.1f}, BOLL位置: {stock['BOLL']}, 成交量比: {stock['成交量比']:.2f}\n")
            f.write("-"*30 + "\n")

        print(f"\n✅ 优质股票已保存: {output_file2}")
    print(f"   找到 {len(quality_stocks_filtered)} 只优质股票（最低综合评分={threshold:.4f}）")

    if len(quality_stocks_filtered) > 0:
        print(f"\n🎯 今日优质股票列表 (前{len(quality_stocks_filtered)}名)：")
        print("="*130)
        print(f"{'股票代码':<10} {'股票名称':<12} {'涨幅%':<8} {'综合评分':<10} {'短期评分':<10} {'长期评分':<10} {'总市值(亿)':<12} {'换手率(%)':<10} {'市盈率(动)':<12} {'所属行业':<15}")
        print("-"*130)
        for stock in quality_stocks_filtered:
            print(f"{stock['代码']:<10} {stock['名称']:<12} {stock['涨幅']:<8.2f} {stock['综合评分']:<10.4f} {stock['短期评分']:<10.4f} {stock['长期评分']:<10.4f} {stock['总市值']:<12.2f} {stock['换手率']:<10.2f} {stock['市盈率(动)']:<12.2f} {stock['行业']:<15}")

        # ========== 第五步：结合分析给出投资建议 ==========
        print("\n   投资建议 (基于模型评分、技术指标和基本面):")
        for stock in quality_stocks_filtered:
            code = stock['代码']
            name = stock['名称']
            comprehensive_score = stock['综合评分']
            short_term_score = stock['短期评分']
            long_term_score = stock['长期评分']
            change_percent = stock['涨幅']
            market_cap = stock['总市值']
            turnover_rate = stock['换手率']
            pe_ratio = stock['市盈率(动)']
            industry = stock['行业']
            macd = stock['MACD']
            rsi = stock['RSI']
            boll = stock['BOLL']
            vol_ratio = stock['成交量比']

            # 1. 基本面分析
            profitability = "优秀" if pe_ratio > 0 and pe_ratio < 15 else "良好" if pe_ratio < 30 else "一般"
            growth_potential = "高" if market_cap < 500 and turnover_rate > 5 else "中" if market_cap < 1000 else "低"
            debt_level = "健康" if market_cap > 100 else "一般"  # 简化评估
            
            # 2. 技术面分析
            macd_signal = "金叉" if macd == 1 else "死叉" if macd == -1 else "中性"
            rsi_signal = "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性"
            boll_signal = "上轨" if boll == 1 else "下轨" if boll == -1 else "中轨"
            volume_signal = "放量" if vol_ratio > 1.2 else "缩量" if vol_ratio < 0.8 else "平量"
            
            # 3. 综合判断和建议
            print(f"\n   股票代码: {code} ({name})")
            print(f"     所属行业: {industry}")
            print(f"     综合评分: {comprehensive_score:.4f} | 短期评分: {short_term_score:.4f} | 长期评分: {long_term_score:.4f}")
            print(f"     基本面: 盈利能力-{profitability}, 成长潜力-{growth_potential}, 负债水平-{debt_level}")
            print(f"     技术面: MACD-{macd_signal}, RSI-{rsi_signal}({rsi:.1f}), BOLL-{boll_signal}, 成交量-{volume_signal}({vol_ratio:.2f})")
            
            # 投资建议 - 根据评分和技术指标
            # 短期策略 (1-5个交易日)
            short_term_recommendation = ""
            if short_term_score > 0.7:
                if macd == 1 and rsi < 70 and boll != 1:
                    short_term_recommendation = "强烈买入"
                elif macd == 1 or rsi < 30:
                    short_term_recommendation = "买入"
                else:
                    short_term_recommendation = "谨慎买入"
            elif short_term_score > 0.5:
                short_term_recommendation = "观望"
            else:
                short_term_recommendation = "回避"
                
            # 长期策略 (1-6个月)
            long_term_recommendation = ""
            if long_term_score > 0.7:
                if pe_ratio < 30 and market_cap > 50:
                    long_term_recommendation = "强烈买入"
                elif pe_ratio < 50:
                    long_term_recommendation = "买入"
                else:
                    long_term_recommendation = "谨慎买入"
            elif long_term_score > 0.5:
                long_term_recommendation = "观望"
            else:
                long_term_recommendation = "回避"
                
            print(f"     短期策略(1-5天): {short_term_recommendation}")
            print(f"     长期策略(1-6月): {long_term_recommendation}")
            
            # 具体操作建议
            print(f"     具体操作:")
            if short_term_recommendation in ["强烈买入", "买入"]:
                print(f"       - 短期: 可在当前价位买入，目标涨幅5-8%，止损设在-3%")
                if rsi > 70:
                    print(f"       - 注意: RSI({rsi:.1f})已进入超买区，可等待回调介入")
            elif short_term_recommendation == "谨慎买入":
                print(f"       - 短期: 可轻仓参与，严格设置止损-3%，快进快出")
            
            if long_term_recommendation in ["强烈买入", "买入"]:
                print(f"       - 长期: 可分批建仓，关注季度财报，目标持有3-6个月")
                if pe_ratio > 30:
                    print(f"       - 注意: 市盈率({pe_ratio:.1f})偏高，等待回调至合理区间")
            elif long_term_recommendation == "谨慎买入":
                print(f"       - 长期: 可小仓位布局，关注行业政策和基本面变化")
            
            if short_term_recommendation == "回避" and long_term_recommendation == "回避":
                print("       - 暂无合适操作策略，建议观望")
                
            print("-" * 70)

    else:
        print("\n⚠️ 今日没有找到符合条件的优质股票")
        print("   可能原因：")
        print("   1. 市场整体表现不佳，涨幅不足")
        print("   2. 数据获取不完整或质量不佳")
        print("   3. 神经网络模型需要更多数据或优化")

    # ========== 第四步：关联规则挖掘 ==========
    # 在这里调用关联规则挖掘函数
    perform_association_rule_mining(df_for_scoring.copy()) # 传入原始数值的df副本

    print("\n" + "="*60)
    print("✅ 程序执行完成！")
    print("="*60)


if __name__ == "__main__":
    main()
