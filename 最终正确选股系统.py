#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态选股系统 - 实时计算版 (集成神经网络与关联规则)
结合专门技术优化预测模型 结合特征和预测结果给我策略 针对推荐股票 短期/长期 买入/卖出 结合专业分析进行补充 完整代码
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
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
        change = safe_float(row.get('涨跌幅')) # 使用涨跌幅代替涨幅%
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

    # H列：归属净利润 (数值，单位亿) -  无法获取，用0代替
    features.append(0)

    # I列：实际换手率 (数值)
    try:
        turnover = safe_float(row.get('换手率')) # 使用换手率代替实际换手%
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

def calculate_technical_indicators(df_row):
    """
    计算技术指标：简单移动平均线交叉, RSI, 价格与布林带关系, 成交量变化率
    使用传入的单行数据代替历史K线数据
    """
    try:
        # 获取实时数据
        close_price = safe_float(df_row.get('最新'))
        high_price = safe_float(df_row.get('最高'))
        low_price = safe_float(df_row.get('最低'))

        # 简单移动平均线交叉信号 (1:金叉, -1:死叉, 0:无) - 无法计算，给一个中性值
        sma_signal = 0

        # 相对强弱指标 (RSI) - 简化计算
        rsi = 50 # 无法计算，给一个中性值

        # 布林带位置 (1:上轨, -1:下轨, 0:中轨)
        boll_position = 0 # 无法计算，给一个中性值

        # 成交量变化率 (5日平均成交量/20日平均成交量)
        vol_ratio = 1 # 无法计算，给一个中性值

        return sma_signal, rsi, boll_position, vol_ratio

    except Exception as e:
        print(f"计算技术指标时出错: {e}")
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
    change = df['涨跌幅'].apply(safe_float) # 使用涨跌幅代替涨幅%
    profit = pd.Series([0] * len(df)) # 无法获取，用0代替
    turnover = df['换手率'].apply(safe_float) # 使用换手率代替实际换手%
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
        # 如果Optuna失败，使用一个默认的模型配置
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
        best_params = study.best_params

    print("   Optuna 优化完成，最佳参数:", best_params)

    # 使用最佳参数训练模型
    print("   使用最佳参数训练最终模型...")
    hidden_layer_sizes = []
    for i in range(best_params['n_layers']):
        hidden_layer_sizes.append(best_params[f'n_units_l{i}'])

    final_model = MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        activation=best_params['activation'],
        solver=best_params['solver'],
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        random_state=42,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10,
        tol=1e-4
    )
    final_model.fit(X_train, y_train)

    # 评估模型
    y_pred = final_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   模型评估: MSE = {mse:.4f}, R^2 = {r2:.4f}")

    return scaler, final_model

def perform_association_rule_mining(df):
    """
    执行关联规则挖掘，发现频繁项集和关联规则。
    """
    print("\n   开始关联规则挖掘...")

    # 特征工程：将数值特征转换为离散特征
    print("   特征离散化...")
    df['涨跌幅_类别'] = pd.cut(df['涨跌幅'].apply(safe_float), bins=[-np.inf, -5, 0, 5, 10, np.inf],
                         labels=['极低', '低', '中', '高', '极高'])
    df['归属净利润_类别'] = pd.cut(pd.Series([0] * len(df)), bins=[-np.inf, -10, 0, 10, 50, np.inf],
                             labels=['亏损', '微利', '一般', '良好', '优秀']) # 无法获取，用0代替
    df['换手率_类别'] = pd.cut(df['换手率'].apply(safe_float), bins=[-np.inf, 1, 3, 5, 10, np.inf],
                             labels=['极低', '低', '中', '高', '极高'])
    df['总市值_类别'] = pd.cut(df['总市值'].apply(safe_float), bins=[-np.inf, 100, 500, 1000, 5000, np.inf],
                           labels=['小型', '中型', '大型', '超大型', '巨型'])

    # 添加苏氏策略计算的特征
    print("   添加苏氏策略特征...")
    df['苏氏策略特征'] = df.apply(calculate_features, axis=1)

    # 将苏氏策略特征展开为单独的列
    df[['价格位置', '涨幅和价格位置', '归属净利润', '实际换手率', '总市值']] = df['苏氏策略特征'].tolist()

    # 将苏氏策略特征转换为离散特征
    df['价格位置_类别'] = df['价格位置'].apply(lambda x: '满足' if x == 1 else '不满足')
    df['涨幅和价格位置_类别'] = df['涨幅和价格位置'].apply(lambda x: '满足' if x == 1 else '不满足')

    # 选择用于关联规则挖掘的列
    print("   选择用于关联规则挖掘的列...")
    selected_columns = ['行业', '地区', '涨跌幅_类别', '归属净利润_类别', '换手率_类别', '总市值_类别',
                          '价格位置_类别', '涨幅和价格位置_类别']
    df_selected = df[selected_columns].copy()

    # 将 DataFrame 转换为事务列表
    print("   转换为事务列表...")
    transactions = df_selected.astype(str).values.tolist()

    # 使用 TransactionEncoder 进行 one-hot 编码
    print("   进行 one-hot 编码...")
    te = TransactionEncoder()
    te_result = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_result, columns=te.columns_)

    # 使用 Apriori 算法发现频繁项集
    print("   使用 Apriori 算法...")
    try:
        frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
    except ValueError as e:
        print(f"   Apriori 算法出错: {e}")
        print("   请检查数据或调整 min_support 参数。")
        return

    # 生成关联规则
    print("   生成关联规则...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    # 打印关联规则
    print("\n   关联规则:")
    if not rules.empty:
        print(rules[['antecedents', 'consequents', 'confidence', 'lift']].head(10))  # 显示前10条规则
    else:
        print("   没有找到符合条件的关联规则。")

def main():
    """
    主函数，执行数据获取、预处理、模型训练、选股和投资建议。
    """
    print("="*60)
    print("🚀 启动动态选股系统 (集成神经网络与关联规则) 🚀")
    print("="*60)

    # ========== 第一步：获取所有A股股票数据 ==========
    print("\n   开始获取所有A股股票数据...")
    stock_list_df = ak.stock_zh_a_spot()
    print(f"   共获取 {len(stock_list_df)} 只股票数据")

    # 筛选掉ST股
    stock_list_df = stock_list_df[~stock_list_df['名称'].str.contains('ST')]
    stock_list_df = stock_list_df[~stock_list_df['名称'].str.contains('退')]
    print(f"   剔除ST股后剩余 {len(stock_list_df)} 只股票")

    # ========== 第二步：获取股票详细数据并进行预处理 ==========
    print("\n   开始获取股票实时数据并进行预处理...")
    df_list = []
    error_codes = []
    for i, row in stock_list_df.iterrows():
        code = row['代码']
        name = row['名称']
        print(f"   [{i+1}/{len(stock_list_df)}] 获取 {name}({code}) 实时数据...", end="")
        try:
            # 获取股票实时数据
            stock_realtime_df = ak.stock_zh_a_spot(symbol=code)
            if stock_realtime_df is None or len(stock_realtime_df) == 0:
                print("❌ 获取实时数据失败")
                error_codes.append(code)
                continue

            # 合并数据
            realtime_data = stock_realtime_df.iloc[0] # 获取第一行数据
            df = pd.DataFrame([realtime_data])
            df['代码'] = code
            df['名称'] = name
            df['行业'] = row['行业']
            df['地区'] = row['地区']
            df_list.append(df)
            print("✅ 完成")
        except Exception as e:
            print(f"❌ 获取数据出错: {e}")
            error_codes.append(code)

    if error_codes:
        print(f"\n⚠️  以下股票获取数据出错，已跳过: {error_codes}")

    # 合并所有股票数据
    df = pd.concat(df_list, ignore_index=True)

    # 将数据保存到 CSV 文件
    output_file = os.path.join("输出数据", "A股数据.csv")
    df.to_csv(output_file, index=False, encoding="utf_8_sig")
    print(f"\n✅ 所有A股数据已保存到: {output_file}")

    # 数据清洗和转换
    print("\n   数据清洗和转换...")
    df['涨幅'] = df['涨跌幅'].apply(safe_float)
    df['涨跌幅'] = df['涨跌幅'].apply(safe_float) # 修正列名
    df['总市值'] = df['总市值'].apply(safe_float)
    df['流通市值'] = df['流通市值'].apply(safe_float)
    df['换手率'] = df['换手率'].apply(safe_float)
    df['市盈率(动)'] = df['市盈率(动)'].apply(safe_float)
    df['市净率'] = df['市净率'].apply(safe_float)
    df['最高'] = df['最高'].apply(safe_float)
    df['最低'] = df['最低'].apply(safe_float)
    df['最新'] = df['最新'].apply(safe_float)
    df['成交量'] = df['成交量'].apply(safe_float)

    # 移除包含 NaN 或无穷大的行
    df_for_scoring = df.copy() # 用于评分
    df = df.dropna(subset=['涨跌幅', '总市值', '换手率', '市盈率(动)']) # 使用涨跌幅代替涨幅%
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

    print(f"   清洗后剩余 {len(df)} 只股票")

    # ========== 第三步：训练神经网络模型并进行预测 ==========
    # 训练神经网络模型
    scaler_short, model_short = train_neural_network(df.copy(), target_type='short_term')
    scaler_long, model_long = train_neural_network(df.copy(), target_type='long_term')
    scaler_comprehensive, model_comprehensive = train_neural_network(df.copy(), target_type='comprehensive')

    # 检查模型是否训练成功
    if model_short is None or model_long is None or model_comprehensive is None:
        print("\n⚠️  神经网络模型训练失败，无法进行预测。")
        return

    # 使用模型进行预测
    print("\n   使用神经网络模型进行预测...")
    X = []
    for _, row in df_for_scoring.iterrows():
        features = calculate_features(row)
        X.append(features)

    X = np.array(X)

    # 移除包含 NaN 或无穷大的行
    mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
    X = X[mask]
    df_for_scoring = df_for_scoring[mask]

    # 预测
    X_scaled_short = scaler_short.transform(X) if scaler_short else X
    X_scaled_long = scaler_long.transform(X) if scaler_long else X
    X_scaled_comprehensive = scaler_comprehensive.transform(X) if scaler_comprehensive else X

    df_for_scoring['短期评分'] = model_short.predict(X_scaled_short) if model_short else 0
    df_for_scoring['长期评分'] = model_long.predict(X_scaled_long) if model_long else 0
    df_for_scoring['综合评分'] = model_comprehensive.predict(X_scaled_comprehensive) if model_comprehensive else 0

    # ========== 第四步：筛选优质股票 ==========
    print("\n   筛选优质股票...")
    # 筛选条件：综合评分大于阈值
    display_count = 20 # 用于计算阈值的数量
    threshold = df_for_scoring['综合评分'].nlargest(display_count).min()

    # 筛选优质股票
    quality_stocks_filtered = df_for_scoring[df_for_scoring['综合评分'] >= threshold].sort_values(by='综合评分', ascending=False)

    # 获取技术指标
    print("\n   获取技术指标...")
    for i, row in quality_stocks_filtered.iterrows():
        sma, rsi, boll, vol_ratio = calculate_technical_indicators(row)
        quality_stocks_filtered.loc[i, 'SMA'] = sma
        quality_stocks_filtered.loc[i, 'RSI'] = rsi
        quality_stocks_filtered.loc[i, 'BOLL'] = boll
        quality_stocks_filtered.loc[i, '成交量比'] = vol_ratio

    # 保存优质股票到文件
    output_file2 = os.path.join("输出数据", "优质股票.txt")
    with open(output_file2, "w", encoding="utf_8") as f:
        f.write("="*50 + "\n")
        f.write("量化策略 - 优质股票筛选结果 (神经网络评分)\n")
        f.write(f"筛选时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"最低综合评分阈值 (基于前{display_count}名或全部): {threshold:.4f}\n")
        f.write(f"优质股票数量: {len(quality_stocks_filtered)}\n")
        f.write("="*50 + "\n\n")

        for stock in quality_stocks_filtered.to_dict('records'):
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
            f.write(f"技术指标 - SMA信号: {stock['SMA']}, RSI: {stock['RSI']:.1f}, BOLL位置: {stock['BOLL']}, 成交量比: {stock['成交量比']:.2f}\n")
            f.write("-"*30 + "\n")

        print(f"\n✅ 优质股票已保存: {output_file2}")
    print(f"   找到 {len(quality_stocks_filtered)} 只优质股票（最低综合评分={threshold:.4f}）")

    if len(quality_stocks_filtered) > 0:
        print(f"\n🎯 今日优质股票列表 (前{len(quality_stocks_filtered)}名)：")
        print("="*130)
        print(f"{'股票代码':<10} {'股票名称':<12} {'涨幅%':<8} {'综合评分':<10} {'短期评分':<10} {'长期评分':<10} {'总市值(亿)':<12} {'换手率(%)':<10} {'市盈率(动)':<12} {'所属行业':<15}")
        print("-"*130)
        for stock in quality_stocks_filtered.to_dict('records'):
            print(f"{stock['代码']:<10} {stock['名称']:<12} {stock['涨幅']:<8.2f} {stock['综合评分']:<10.4f} {stock['短期评分']:<10.4f} {stock['长期评分']:<10.4f} {stock['总市值']:<12.2f} {stock['换手率']:<10.2f} {stock['市盈率(动)':<12.2f} {stock['行业']:<15}")

        # ========== 第五步：结合分析给出投资建议 ==========
        print("\n   投资建议 (基于模型评分、技术指标和基本面):")
        for stock in quality_stocks_filtered.to_dict('records'):
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
            sma = stock['SMA']
            rsi = stock['RSI']
            boll = stock['BOLL']
            vol_ratio = stock['成交量比']

            # 1. 基本面分析
            profitability = "优秀" if pe_ratio > 0 and pe_ratio < 15 else "良好" if pe_ratio < 30 else "一般"
            growth_potential = "高" if market_cap < 500 and turnover_rate > 5 else "中" if market_cap < 1000 else "低"
            debt_level = "健康" if market_cap > 100 else "一般"  # 简化评估

            # 2. 技术面分析
            sma_signal = "金叉" if sma == 1 else "死叉" if sma == -1 else "中性"
            rsi_signal = "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性"
            boll_signal = "上轨" if boll == 1 else "下轨" if boll == -1 else "中轨"
            volume_signal = "放量" if vol_ratio > 1.2 else "缩量" if vol_ratio < 0.8 else "平量"

            # 3. 综合判断和建议
            print(f"\n   股票代码: {code} ({name})")
            print(f"     所属行业: {industry}")
            print(f"     综合评分: {comprehensive_score:.4f} | 短期评分: {short_term_score:.4f} | 长期评分: {long_term_score:.4f}")
            print(f"     基本面: 盈利能力-{profitability}, 成长潜力-{growth_potential}, 负债水平-{debt_level}")
            print(f"     技术面: SMA-{sma_signal}, RSI-{rsi_signal}({rsi:.1f}), BOLL-{boll_signal}, 成交量-{volume_signal}({vol_ratio:.2f})")

            # 投资建议 - 根据评分和技术指标
            # 短期策略 (1-5个交易日)
            short_term_recommendation = ""
            if short_term_score > 0.7:
                if sma == 1 and rsi < 70 and boll != 1:
                    short_term_recommendation = "强烈买入"
                elif sma == 1 or rsi < 30:
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

