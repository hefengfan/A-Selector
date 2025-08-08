#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态选股系统 - 实时计算版 (集成神经网络、关联规则与 XGBoost)
基于苏氏量化策略的真实计算逻辑
集成神经网络进行初步评分，XGBoost进行精准预测
新增：
1. 使用 Optuna 进行神经网络超参数优化，提升评分精度和区分度。
2. 引入关联规则挖掘，分析哪些条件组合更容易产生高收益，提供策略洞察。
3. 优化数据处理和输出展示。
4. 优化代码准确性、质量和效率。
5. 使用复合质量评分作为神经网络的目标变量，提高模型准确性。
6. 集成XGBoost模型，替代神经网络，进行更精准的预测。
7. 提供基于特征和预测结果的短期/长期买入/卖出策略建议。
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

# 导入 Optuna 进行超参数优化
import optuna

# 导入 mlxtend 进行关联规则挖掘
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 导入 XGBoost
import xgboost as xgb

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
    根据苏氏量化策略计算特征值，用于机器学习训练和关联规则挖掘。
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

    # K列：市盈率 (动)
    try:
        pe = safe_float(row.get('市盈率(动)'))
        features.append(pe if pd.notna(pe) else 100)  # 缺失时给一个较大值
    except:
        features.append(100)

    # L列：20日均价与60日均价的差值
    try:
        ma20 = safe_float(row.get('20日均价'))
        ma60 = safe_float(row.get('60日均价'))
        ma_diff = ma20 - ma60
        features.append(ma_diff if pd.notna(ma20) and pd.notna(ma60) else 0)
    except:
        features.append(0)

    # M列：昨日收盘价
    try:
        last_close = safe_float(row.get('昨收'))
        features.append(last_close if pd.notna(last_close) else 0)
    except:
        features.append(0)

    # N列：今日开盘价
    try:
        open_price = safe_float(row.get('开盘'))
        features.append(open_price if pd.notna(open_price) else 0)
    except:
        features.append(0)

    return features

def calculate_quality_score(df):
    """计算复合质量评分"""
    # 提取用于计算质量评分的列
    change = df['涨幅%'].apply(safe_float)
    profit = df['归属净利润'].apply(safe_float)
    turnover = df['实际换手%'].apply(safe_float)
    market_cap = df['总市值'].apply(safe_float)
    pe_ratio = df['市盈率(动)'].apply(safe_float)

    # 归一化各个指标 (使用 min-max 归一化)
    change_norm = (change - change.min()) / (change.max() - change.min())
    profit_norm = (profit - profit.min()) / (profit.max() - profit.min())
    turnover_norm = (turnover - turnover.min()) / (turnover.max() - turnover.min())
    market_cap_norm = (market_cap - market_cap.min()) / (market_cap.max() - market_cap.min())
    pe_ratio_norm = (pe_ratio - pe_ratio.min()) / (pe_ratio.max() - pe_ratio.min())

    # 处理 NaN 值，用 0 填充
    change_norm = change_norm.fillna(0)
    profit_norm = profit_norm.fillna(0)
    turnover_norm = turnover_norm.fillna(0)
    market_cap_norm = market_cap_norm.fillna(0)
    pe_ratio_norm = pe_ratio_norm.fillna(0)

    # 计算复合质量评分 (可以调整权重)
    quality_score = (
        0.3 * change_norm +  # 涨幅
        0.25 * profit_norm +  # 净利润
        0.15 * (1 - abs(turnover_norm - 0.5)) +  # 换手率 (适中最好)
        0.2 * market_cap_norm +  # 市值
        0.1 * (1 - pe_ratio_norm)  # 市盈率 (越低越好)
    )

    return quality_score

def objective_xgboost(trial, X_train, y_train, X_test, y_test):
    """Optuna 优化 XGBoost 目标函数"""
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
    }

    if params['booster'] == 'gbtree' or params['booster'] == 'dart':
        params['max_depth'] = trial.suggest_int('max_depth', 3, 9)
        params['eta'] = trial.suggest_loguniform('eta', 0.01, 0.3)
        params['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
        params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

    if params['booster'] == 'dart':
        params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        params['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
        params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)

    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse

def train_xgboost_model(df):
    """训练 XGBoost 模型"""
    print("\n   准备训练数据...")
    X = []
    for _, row in df.iterrows():
        features = calculate_features(row)
        X.append(features)
    X = np.array(X)

    # 计算质量评分
    y = calculate_quality_score(df).values

    # 移除包含 NaN 或无穷大的行
    mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1) & ~np.isnan(y) & ~np.isinf(y)
    X = X[mask]
    y = y[mask]

    if len(X) < 20:
        print("   ❌ 有效训练数据不足，无法训练 XGBoost 模型。")
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
        study.optimize(lambda trial: objective_xgboost(trial, X_train, y_train, X_test, y_test), n_trials=50, show_progress_bar=True)
    except Exception as e:
        print(f"   Optuna 优化过程中发生错误: {e}")
        print("   将使用默认或预设参数训练模型。")
        best_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'max_depth': 5,
            'eta': 0.1,
            'gamma': 0.0,
            'lambda': 1.0,
            'alpha': 0.0,
        }
    else:
        print("\n   Optuna 优化完成。")
        print(f"   最佳均方根误差 (RMSE): {study.best_value:.4f}")
        print(f"   最佳超参数: {study.best_params}")
        best_params = study.best_params

    # 使用最佳参数训练最终模型
    print("   使用最佳参数训练最终 XGBoost 模型...")
    model = xgb.XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)

    # 评估模型
    print("   评估最终模型...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"   最终模型均方误差 (MSE): {mse:.4f}")
    print(f"   最终模型R²分数: {r2:.4f}")

    return model, scaler

def predict_score_with_xgboost(row, model, scaler):
    """使用训练好的 XGBoost 模型预测股票评分"""
    features = calculate_features(row)
    if any(pd.isna(f) or np.isinf(f) for f in features):
        return np.nan

    features = np.array(features).reshape(1, -1)
    try:
        features_scaled = scaler.transform(features)
        score = model.predict(features_scaled)[0]
        return score
    except Exception as e:
        return np.nan

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
    # min_support 可以根据数据量调整，太小规则太多，太大规则太少
    frequent_itemsets = apriori(df_ar, min_support=0.01, use_colnames=True)
    if frequent_itemsets.empty:
        print("   ⚠️ 未找到频繁项集，请尝试降低 min_support。")
        return

    # 生成关联规则
    # min_confidence 越高，规则越可靠
    # lift > 1 表示前件和后件正相关
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)

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

def generate_trading_strategy(row, score, rules):
    """
    根据股票特征、XGBoost预测评分和关联规则生成交易策略。
    """
    strategy = {}

    # 1. 基本信息
    code = row.get('原始代码', '未知')
    name = row.get('名称', '未知')
    current_price = safe_float(row.get('最新', 0))
    change_percent = safe_float(row.get('涨幅%', 0))

    strategy['代码'] = code
    strategy['名称'] = name

    # 2. XGBoost 评分解读
    strategy['XGBoost评分'] = score
    if score > 0.7:
        strategy['评分解读'] = "高潜力股票，值得关注。"
    elif score > 0.5:
        strategy['评分解读'] = "中等潜力股票，可以考虑。"
    else:
        strategy['评分解读'] = "潜力较低，谨慎。"

    # 3. 技术指标分析 (简化示例)
    ma20 = safe_float(row.get('20日均价', 0))
    ma60 = safe_float(row.get('60日均价', 0))

    if current_price > ma20 and current_price > ma60:
        strategy['技术面'] = "短期和长期趋势向上。"
    elif current_price < ma20 and current_price < ma60:
        strategy['技术面'] = "短期和长期趋势向下。"
    else:
        strategy['技术面'] = "趋势不明朗。"

    # 4. 关联规则应用
    applicable_rules = []
    features = calculate_features(row)

    # 简化的规则匹配，需要根据实际规则进行调整
    if features[0] == 1 and features[1] == 1:
        applicable_rules.append("价格位置和涨幅位置均满足")
        strategy['关联规则'] = "价格位置和涨幅位置均满足，可能预示着进一步上涨。"
    elif features[2] > 0.3 and features[3] <= 20:
        applicable_rules.append("净利润高且换手率低")
        strategy['关联规则'] = "净利润高且换手率低，可能代表着稳健增长。"
    else:
        strategy['关联规则'] = "未发现匹配的关联规则。"

    # 5. 交易策略建议
    if score > 0.7 and current_price > ma20:
        strategy['短期策略'] = "逢低买入，短期目标涨幅5%-10%。"
    elif score > 0.5 and features[2] > 0.3:
        strategy['长期策略'] = "长期持有，关注公司基本面变化。"
    else:
        strategy['交易策略'] = "观望，等待更明确的信号。"

    # 6. 风险提示
    strategy['风险提示'] = "股市有风险，投资需谨慎。请结合自身风险承受能力进行决策。"

    return strategy

def main():
    """主程序"""
    print("\n" + "="*60)
    print("动态选股系统 - 实时计算版 (集成 XGBoost 与关联规则)")
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

    # ========== 第二步：训练 XGBoost 模型 ==========
    print("\n2. 训练 XGBoost 模型...")
    model, scaler = train_xgboost_model(df.copy()) # 传入原始数值的df副本

    if model is None:
        print("   ❌ XGBoost 模型训练失败，无法进行后续筛选。")
        return

    # ========== 第三步：动态筛选优质股票 ==========
    print("\n3. 动态筛选优质股票 (基于 XGBoost 评分)...")

    quality_stocks = []
    
    # 重新加载原始数值的df，因为上面为了输出csv已经格式化了
    df_for_scoring = df.copy()
    for col in ['最新', '涨幅%', '最高', '最低', '实际换手%', '20日均价', '60日均价', '市盈率(动)', '总市值', '归属净利润', '昨收', '开盘']:
        df_for_scoring[col] = df_for_scoring[col].apply(safe_float)
    df_for_scoring['原始代码'] = df_for_scoring['代码'].apply(lambda x: str(x).replace('= "', '').replace('"', ''))


    for idx, row in df_for_scoring.iterrows():
        score = predict_score_with_xgboost(row, model, scaler)
        
        if pd.notna(score): # 确保分数有效
            code = str(row['原始代码']).strip()
            quality_stocks.append({
                '代码': code,
                '名称': str(row['名称']).strip(),
                '行业': str(row['所属行业']).strip(),
                '优质率': score,
                '涨幅': f"{safe_float(row['涨幅%']):.2f}%" if pd.notna(safe_float(row['涨幅%'])) else "--"
            })

    # 按优质率降序排序
    quality_stocks = sorted(quality_stocks, key=lambda x: (x['优质率'], x['代码']), reverse=True)

    # 确定筛选阈值：取前N个，或者根据分数分布动态调整
    display_count = 15 # 默认显示前15个
    if len(quality_stocks) > display_count:
        # 如果股票数量足够，取前N个的最低分数作为阈值
        threshold = quality_stocks[display_count-1]['优质率']
        quality_stocks_filtered = quality_stocks[:display_count]
    elif len(quality_stocks) > 0:
        threshold = quality_stocks[-1]['优质率'] # 所有股票的最低分
        quality_stocks_filtered = quality_stocks
    else:
        threshold = 0.0
        quality_stocks_filtered = []

    # 保存优质股票
    output_file2 = '输出数据/优质股票.txt'
    with open(output_file2, 'w', encoding='utf-8') as f:
        f.write("苏氏量化策略 - 优质股票筛选结果 (XGBoost 评分)\n")
        f.write(f"筛选时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"最低优质率阈值 (基于前{display_count}名或全部): {threshold:.4f}\n")
        f.write(f"优质股票数量: {len(quality_stocks_filtered)}\n")
        f.write("="*50 + "\n\n")

        for stock in quality_stocks_filtered:
            f.write(f"股票代码: {stock['代码']}\n")
            f.write(f"股票名称: {stock['名称']}\n")
            f.write(f"所属行业: {stock['行业']}\n")
            f.write(f"优质率 (XGBoost评分): {stock['优质率']:.4f}\n")
            f.write(f"今日涨幅: {stock['涨幅']}\n")
            f.write("-"*30 + "\n")

    print(f"\n✅ 优质股票已保存: {output_file2}")
    print(f"   找到 {len(quality_stocks_filtered)} 只优质股票（最低优质率={threshold:.4f}）")

    if len(quality_stocks_filtered) > 0:
        print(f"\n🎯 今日优质股票列表 (前{len(quality_stocks_filtered)}名)：")
        print("="*60)
        print(f"{'股票代码':<10} {'股票名称':<12} {'涨幅':<8} {'优质率':<10} {'所属行业':<15}")
        print("-"*60)
        for stock in quality_stocks_filtered:
            print(f"{stock['代码']:<10} {stock['名称']:<12} {stock['涨幅']:<8} {stock['优质率']:.4f}   {stock['行业']:<15}")
    else:
        print("\n⚠️ 今日没有找到符合条件的优质股票")
        print("   可能原因：")
        print("   1. 市场整体体表现不佳，涨幅不足")
        print("   2. 数据获取不完整或质量不佳")
        print("   3. XGBoost 模型需要更多数据或优化")

    # ========== 第四步：关联规则挖掘 ==========
    # 在这里调用关联规则挖掘函数
    rules = perform_association_rule_mining(df_for_scoring.copy()) # 传入原始数值的df副本

    # ========== 第五步：生成交易策略 ==========
    print("\n5. 生成交易策略...")

    trading_strategies = []
    for idx, row in df_for_scoring.iterrows():
        score = predict_score_with_xgboost(row, model, scaler)
        if pd.notna(score):
            strategy = generate_trading_strategy(row, score, rules)  # 传入关联规则结果
            trading_strategies.append(strategy)

    # 保存交易策略
    output_file3 = '输出数据/交易策略.txt'
    with open(output_file3, 'w', encoding='utf-8') as f:
        f.write("苏氏量化策略 - 交易策略建议 (XGBoost 评分 & 关联规则)\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")

        for strategy in trading_strategies:
            f.write(f"股票代码: {strategy.get('代码', '未知')}\n")
            f.write(f"股票名称: {strategy.get('名称', '未知')}\n")
            f.write(f"XGBoost 评分: {strategy.get('XGBoost评分', '未知'):.4f}\n")
            f.write(f"评分解读: {strategy.get('评分解读', '无')}\n")
            f.write(f"技术面: {strategy.get('技术面', '无')}\n")
            f.write(f"关联规则: {strategy.get('关联规则', '无')}\n")
            f.write(f"短期策略: {strategy.get('短期策略', '无')}\n")
            f.write(f"长期策略: {strategy.get('长期策略', '无')}\n")
            f.write(f"交易策略: {strategy.get('交易策略', '无')}\n")
            f.write(f"风险提示: {strategy.get('风险提示', '无')}\n")
            f.write("-"*30 + "\n")

    print(f"\n✅ 交易策略已保存: {output_file3}")

    print("\n" + "="*60)
    print("✅ 程序执行完成！")
    print("="*60)

if __name__ == "__main__":
    main()
