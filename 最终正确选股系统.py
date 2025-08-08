#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态选股系统 - 根据每天实时数据筛选
基于苏氏量化策略的真实计算逻辑
集成神经网络进行精准评分 (Scikit-learn + Optuna)
集成关联规则挖掘 (Apriori)
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 导入 Scikit-learn 和 Optuna 相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import optuna

# 导入关联规则挖掘库
from mlxtend.frequent_patterns import apriori, association_rules

# 清除代理设置
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['ALL_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

# 定义需要转换为数值的列
NUMERIC_COLS = [
    '最新', '涨幅%', '最高', '最低', '实际换手%', '20日均价', '60日均价',
    '归属净利润', '总市值', '市盈率(动)', '昨收', '开盘'
]

def clean_and_convert_numeric(df, cols_to_convert):
    """
    将指定列转换为数值类型，并处理缺失值。
    """
    for col in cols_to_convert:
        if col in df.columns:
            # 尝试替换常见的非数值表示
            df[col] = df[col].astype(str).str.replace('--', '0').str.replace(' ', '').str.replace(',', '')

            # 处理亿、万亿等单位
            def parse_value(val):
                if '万亿' in val:
                    return float(val.replace('万亿', '')) * 1_000_000_000_000
                elif '亿' in val:
                    return float(val.replace('亿', '')) * 1_000_000_00
                elif '万' in val:
                    return float(val.replace('万', '')) * 10_000
                return val

            df[col] = df[col].apply(parse_value)

            # 转换为数值，无法转换的设为NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 填充NaN，这里选择0，也可以根据实际情况选择中位数或均值
            df[col] = df[col].fillna(0)
    return df


def calculate_features(row):
    """
    根据苏氏量化策略计算特征值，直接使用数值列
    """
    features = []

    # F列：价格位置条件
    low = row['最低']
    ma60 = row['60日均价']
    ma20 = row['20日均价']
    current = row['最新']

    condition_met_F = 0
    if ma60 > 0 and 0.85 <= low / ma60 <= 1.15:
        condition_met_F = 1
    elif ma20 > 0 and 0.90 <= current / ma20 <= 1.10:
        condition_met_F = 1
    features.append(condition_met_F)

    # G列：涨幅和价格位置
    change = row['涨幅%']
    high = row['最高']
    low_price = row['最低'] # 避免变量名冲突
    current_price = row['最新'] # 避免变量名冲突

    condition_met_G = 0
    if change >= 5.0:
        # 确保 high 和 low_price 是有效数字，避免除以0或NaN
        if high > low_price:
            threshold = high - (high - low_price) * 0.30
            if current_price >= threshold:
                condition_met_G = 1
    features.append(condition_met_G)

    # H列：净利润>=3000万 (0.3亿)
    profit = row['归属净利润']
    features.append(profit)

    # I列：换手率<=20%
    turnover = row['实际换手%']
    features.append(turnover)

    # J列：市值>=300亿
    cap = row['总市值']
    features.append(cap)

    return features


def create_model(trial, input_shape):
    """
    使用 Optuna 建议的超参数创建 Scikit-learn MLPRegressor 模型
    """
    n_layers = trial.suggest_int('n_layers', 1, 4)  # 增加层数
    hidden_layer_sizes = []
    for i in range(n_layers):
        num_units = trial.suggest_int(f'n_units_{i}', 64, 512)  # 增加节点数
        hidden_layer_sizes.append(num_units)

    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic', 'identity']) # 增加激活函数
    solver = trial.suggest_categorical('solver', ['adam', 'lbfgs'])
    alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)  # 调整 alpha 范围
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True) # 调整学习率

    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                         activation=activation,
                         solver=solver,
                         alpha=alpha,
                         learning_rate_init=learning_rate_init,
                         random_state=42,
                         max_iter=500,  # 增加迭代次数
                         early_stopping=True, # 启用早停
                         n_iter_no_change=20, # 增加容忍度
                         tol=1e-4) # 增加收敛容忍度

    return model


def objective(trial, X_train, y_train, X_test, y_test):
    """
    Optuna 优化的目标函数
    """
    model = create_model(trial, X_train.shape[1])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 尝试优化R2分数，因为R2更能体现模型的解释能力
    # Optuna 默认是最小化，所以我们最小化 (1 - R2)
    # 也可以尝试最小化 MSE
    return 1 - r2


def train_neural_network(df):
    """
    训练神经网络模型 (Scikit-learn + Optuna)，预测股票评分
    """

    # 1. 准备训练数据
    print("\n   准备训练数据...")
    X = []
    y = []  # 目标变量：涨幅作为评分的依据

    # 确保所有特征列和目标列都是数值类型且无NaN
    df_for_nn = df.copy()
    # 确保 calculate_features 依赖的列都是数值
    df_for_nn = clean_and_convert_numeric(df_for_nn, NUMERIC_COLS)

    for _, row in df_for_nn.iterrows():
        features = calculate_features(row)
        X.append(features)
        y.append(row['涨幅%']) # 涨幅% 已经通过 clean_and_convert_numeric 处理过

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # 联合清洗 X 和 y，移除包含 NaN 或无穷大的行
    combined_mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1) & \
                    ~np.isnan(y) & ~np.isinf(y)

    X = X[combined_mask]
    y = y[combined_mask]

    if len(X) == 0:
        print("   ❌ 没有有效的训练数据，无法训练神经网络。")
        return None, None, 0 # 返回0作为R2分数

    # 2. 数据预处理
    print("   数据预处理...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3. 划分训练集和测试集
    print("   划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 使用 Optuna 优化超参数
    print("   使用 Optuna 优化超参数...")
    study = optuna.create_study(direction='minimize') # 最小化 (1-R2)

    from functools import partial
    objective_partial = partial(objective, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    try:
        study.optimize(objective_partial, n_trials=30, timeout=180)  # 增加 trials 和 timeout
    except Exception as e:
        print(f"   ⚠️ Optuna 优化过程中发生错误: {e}")
        print("   将使用默认参数或已找到的最佳参数。")
        if not study.trials: # 如果没有成功运行任何trial
            print("   ❌ Optuna 未能完成任何 trial，无法得到最佳参数。")
            return None, None, 0

    # 5. 使用最佳超参数创建模型
    print("   使用最佳超参数创建模型...")
    best_model = create_model(study.best_trial, X_train.shape[1])
    best_model.fit(X_train, y_train)

    # 6. 评估最佳模型
    print("   评估最佳模型...")
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"   均方误差 (MSE): {mse:.4f}")
    print(f"   R^2 Score: {r2:.4f}")

    print("   最佳超参数:")
    print(study.best_params)

    return best_model, scaler, r2


def predict_score_with_nn(row, model, scaler):
    """
    使用训练好的 Scikit-learn 神经网络模型预测股票评分
    """
    # 确保 row 中的特征是数值类型
    # 注意：这里假设 row 已经经过了 clean_and_convert_numeric 处理
    features = calculate_features(row)
    features = np.array(features).reshape(1, -1)  # 转换为二维数组

    # 检查是否有缺失值或无穷值
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        return 0  # 如果有，返回一个默认值

    features_scaled = scaler.transform(features)
    score = model.predict(features_scaled)[0]
    return score


def analyze_association_rules(df):
    """
    使用 Apriori 算法分析股票数据中的关联规则
    """
    print("\n   分析关联规则...")

    # 选择用于关联规则分析的特征列，并定义它们的二值化阈值
    # 这里选择一些有意义的数值特征进行二值化
    # 确保这些列在 df 中是数值类型且无NaN
    df_apriori = df[['涨幅%', '实际换手%', '归属净利润', '总市值']].copy()
    df_apriori = clean_and_convert_numeric(df_apriori, df_apriori.columns.tolist())

    # 将数值特征转换为布尔值 (0 或 1)
    # 定义一些有意义的阈值来创建二值化特征
    df_encoded = pd.DataFrame()
    df_encoded['涨幅_高'] = df_apriori['涨幅%'] > 3.0 # 涨幅大于3%
    df_encoded['换手率_低'] = df_apriori['实际换手%'] < 10.0 # 换手率低于10%
    df_encoded['净利润_高'] = df_apriori['归属净利润'] > 0.5 # 净利润大于0.5亿
    df_encoded['市值_大'] = df_apriori['总市值'] > 500.0 # 市值大于500亿

    # 转换为布尔类型 DataFrame
    df_encoded = df_encoded.astype(bool)

    # 移除全为 False 的行，这些行对关联规则没有贡献
    df_encoded = df_encoded[df_encoded.any(axis=1)]

    if df_encoded.empty:
        print("   ⚠️ 没有足够的二值化数据进行关联规则分析。")
        return pd.DataFrame()

    # 使用 Apriori 算法找到频繁项集
    # 调整 min_support，如果数据量大，可以适当提高
    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

    if frequent_itemsets.empty:
        print("   ⚠️ 没有找到频繁项集。")
        return pd.DataFrame()

    # 生成关联规则
    # 调整 metric 和 min_threshold
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    print(f"   找到 {len(rules)} 条关联规则")
    return rules


def main():
    """主程序"""
    print("\n" + "="*60)
    print("动态选股系统 - 实时计算版")
    print("集成神经网络进行精准评分 (Scikit-learn + Optuna)")
    print("集成关联规则挖掘 (Apriori)")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 创建输出目录
    os.makedirs('输出数据', exist_ok=True)

    # ========== 第一步：获取数据 ==========
    print("\n1. 获取A股数据...")

    df = pd.DataFrame() # 初始化一个空的DataFrame
    # 先尝试获取实时数据
    try:
        print("   尝试获取实时数据...")
        df_realtime = ak.stock_zh_a_spot_em()
        print(f"   ✅ 成功获取 {len(df_realtime)} 只股票的实时数据")

        # 统一列名以匹配参考数据，并进行初步处理
        df_realtime = df_realtime.rename(columns={
            '最新价': '最新',
            '涨跌幅': '涨幅%',
            '换手率': '实际换手%',
            '总市值': '总市值',
            '市盈率-动态': '市盈率(动)' # 注意这里可能需要根据实际数据调整
        })

        # 确保所有关键列存在，如果不存在则创建并填充默认值
        for col in ['20日均价', '60日均价', '所属行业', '归属净利润']:
            if col not in df_realtime.columns:
                df_realtime[col] = ' --'

        df = df_realtime.copy()

    except Exception as e:
        print(f"   ❌ 实时获取失败: {e}")
        print(f"   错误信息: {e}")  # 打印详细错误信息
        print("   使用参考数据作为备选...")

        # 使用参考数据
        try:
            df = pd.read_csv('参考数据/Table.xls', sep='\t', encoding='gbk', dtype=str)
            print(f"   ✅ 从参考文件加载了 {len(df)} 条数据")
            # 移除Excel公式前缀
            df['代码'] = df['代码'].str.replace('= "', '').str.replace('"', '')
        except Exception as e2:
            print(f"   ❌ 无法加载参考数据: {e2}")
            return

    # 保存原始代码，用于后续合并和查找
    df['原始代码'] = df['代码'].copy()

    # 尝试补充均线和财务数据 (如果实时数据缺失)
    try:
        ref_df = pd.read_csv('参考数据/Table.xls', sep='\t', encoding='gbk', dtype=str)
        ref_df['代码'] = ref_df['代码'].str.replace('= "', '').str.replace('"', '') # 清理参考数据代码
        ref_map = ref_df.set_index('代码').to_dict('index')

        # 合并参考数据到主df
        # 使用 merge 更高效和健壮
        df = df.set_index('原始代码').combine_first(ref_df.set_index('代码')).reset_index()
        df = df.rename(columns={'index': '原始代码'}) # 恢复列名

        print(f"   ✅ 补充了 {len(ref_map)} 条参考数据")
    except Exception as e:
        print(f"   ⚠️ 无法补充参考数据: {e}")

    # 对所有需要数值化的列进行清洗和转换
    df = clean_and_convert_numeric(df, NUMERIC_COLS)

    # 格式化代码为Excel公式，便于在Excel中点击
    df['代码'] = df['原始代码'].apply(lambda x: f'= "{str(x)}"')

    # 格式化数值列为字符串，用于最终输出，保留两位小数
    for col in [c for c in NUMERIC_COLS if c in df.columns]:
        df[col] = df[col].apply(lambda x: f" {x:.2f}" if pd.notna(x) else " --")

    # 处理名称
    df['名称'] = df['名称'].apply(lambda x: f" {x}" if pd.notna(x) and not str(x).startswith(' ') else str(x))

    # 添加序号
    df['序'] = range(1, len(df) + 1)
    df['Unnamed: 16'] = '' # 空列

    # 选择输出列
    output_columns = [
        '序', '代码', '名称', '最新', '涨幅%', '最高', '最低',
        '实际换手%', '所属行业', '20日均价', '60日均价',
        '市盈率(动)', '总市值', '归属净利润', '昨收', '开盘', 'Unnamed: 16'
    ]

    # 确保所有输出列都存在
    for col in output_columns:
        if col not in df.columns:
            df[col] = ' --' if col != 'Unnamed: 16' else ''

    final_df = df[output_columns].copy() # 复制一份，避免SettingWithCopyWarning

    # 保存A股数据
    output_file1 = '输出数据/A股数据.csv'
    try:
        final_df.to_csv(output_file1, index=False, encoding='utf-8-sig')
        print(f"\n✅ A股数据已保存: {output_file1}")
    except Exception as e:
        print(f"\n❌ 无法保存 A 股数据: {e}")
        print(f"   错误信息: {e}")

    print(f"   共 {len(final_df)} 只股票")

    # ========== 第二步：训练神经网络 ==========
    print("\n2. 训练神经网络模型...")
    # 传递原始的df，让train_neural_network内部进行数值化和清洗
    model, scaler, r2_score_nn = train_neural_network(df.copy()) # 传递副本

    if model is None:
        print("   ❌ 神经网络训练失败，无法进行后续筛选。")
        return

    # ========== 第三步：分析关联规则 ==========
    print("\n3. 分析关联规则...")
    rules = analyze_association_rules(df.copy()) # 传递副本

    # ========== 第四步：动态筛选优质股票 ==========
    print("\n4. 动态筛选优质股票...")

    # 重新加载或确保 df 包含原始数值数据，以便神经网络评分
    # 这里使用原始的 df (已经过 clean_and_convert_numeric 处理的)
    df_for_scoring = df.copy()
    df_for_scoring = clean_and_convert_numeric(df_for_scoring, NUMERIC_COLS)

    # 创建一个包含所有股票评分的列
    df_for_scoring['神经网络评分'] = df_for_scoring.apply(lambda row: predict_score_with_nn(row, model, scaler), axis=1)

    quality_stocks = []
    # 初始阈值可以根据神经网络评分分布来定，或者先设一个较低的值
    # 比如，取所有股票评分的20%分位数作为初始阈值
    if not df_for_scoring['神经网络评分'].empty:
        initial_threshold = df_for_scoring['神经网络评分'].quantile(0.75) # 取75%分位数
    else:
        initial_threshold = 0.0
    
    threshold = initial_threshold
    print(f"   初始筛选阈值: {threshold:.4f}")

    # 统计
    stats = {'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0}

    for idx, row in df_for_scoring.iterrows():
        score_nn = row['神经网络评分']  # 神经网络评分
        conditions = ""

        # 统计（原始评分方式的统计）
        features = calculate_features(row)
        if features[0] == 1: stats['F'] += 1
        if features[1] == 1: stats['G'] += 1
        if features[2] > 0.3: stats['H'] += 1
        if features[3] <= 20: stats['I'] += 1 # 换手率 <= 20%
        if features[4] >= 300: stats['J'] += 1 # 市值 >= 300亿

        # 综合评分：神经网络评分 + 关联规则加权
        # 这里只是一个示例，你需要根据你的关联规则分析结果来设计加权策略
        score_rules = 0.0  # 初始关联规则评分
        # 示例：如果股票满足某些关联规则，则增加评分
        # 你需要根据你的关联规则分析结果来设计具体的规则判断逻辑
        # 例如：
        # if not rules.empty:
        #     for _, rule_row in rules.iterrows():
        #         antecedent = list(rule_row['antecedents'])
        #         consequent = list(rule_row['consequents'])
        #         # 假设你的规则是 '涨幅_高' -> '净利润_高'
        #         if '涨幅_高' in antecedent and row['涨幅%'] > 3.0 and \
        #            '净利润_高' in consequent and row['归属净利润'] > 0.5:
        #             score_rules += rule_row['confidence'] * 0.1 # 简单加权

        final_score = score_nn + score_rules  # 综合评分

        # 判断是否达标
        if final_score >= threshold:
            code = str(row['原始代码']).strip() # 使用原始代码
            quality_stocks.append({
                '代码': code,
                '名称': str(row['名称']).strip(),
                '行业': str(row['所属行业']).strip(),
                '优质率': final_score,
                '满足条件': conditions,
                '涨幅': str(row['涨幅%']).strip()
            })

    # 打印统计
    total_stocks_evaluated = len(df_for_scoring)
    if total_stocks_evaluated > 0:
        print(f"\n   条件满足统计（共{total_stocks_evaluated}只股票）：")
        print(f"   F列(价格位置): {stats['F']}只 ({stats['F']/total_stocks_evaluated*100:.1f}%)")
        print(f"   G列(涨幅条件): {stats['G']}只 ({stats['G']/total_stocks_evaluated*100:.1f}%)")
        print(f"   H列(净利润>=0.3亿): {stats['H']}只 ({stats['H']/total_stocks_evaluated*100:.1f}%)")
        print(f"   I列(换手率<=20%): {stats['I']}只 ({stats['I']/total_stocks_evaluated*100:.1f}%)")
        print(f"   J列(市值>=300亿): {stats['J']}只 ({stats['J']/total_stocks_evaluated*100:.1f}%)")

    # 按优质率降序排序
    quality_stocks = sorted(quality_stocks, key=lambda x: (x['优质率'], x['代码']), reverse=True)

    # 如果结果太少，尝试降低阈值，或者直接取前N名
    if len(quality_stocks) < 10 and len(df_for_scoring) > 0:
        print(f"\n   ⚠️ 只找到{len(quality_stocks)}只股票，尝试降低阈值并取前12名...")
        # 重新根据所有股票的神经网络评分排序，取前12名
        all_stocks_sorted_by_nn_score = sorted(
            df_for_scoring.to_dict('records'),
            key=lambda x: x['神经网络评分'] if pd.notna(x['神经网络评分']) else -np.inf,
            reverse=True
        )
        
        quality_stocks = []
        for stock_data in all_stocks_sorted_by_nn_score[:12]:
            code = str(stock_data['原始代码']).strip()
            quality_stocks.append({
                '代码': code,
                '名称': str(stock_data['名称']).strip(),
                '行业': str(stock_data['所属行业']).strip(),
                '优质率': stock_data['神经网络评分'],
                '满足条件': "", # 此时条件不明确，清空
                '涨幅': str(stock_data['涨幅%']).strip()
            })
        
        # 重新排序，确保最终结果的优质率是正确的
        quality_stocks = sorted(quality_stocks, key=lambda x: (x['优质率'], x['代码']), reverse=True)
        # 更新阈值，以便报告
        if quality_stocks:
            threshold = quality_stocks[-1]['优质率'] # 此时阈值是第12名的优质率
        else:
            threshold = 0.0


    # 计算最终模型评分 (示例)
    final_model_score = (r2_score_nn * 100 + len(rules) * 0.5) # 神经网络 R^2 乘以100，关联规则数量加权
    # 请根据实际情况调整评分计算方式

    # 保存优质股票
    output_file2 = '输出数据/优质股票.txt'
    try:
        with open(output_file2, 'w', encoding='utf-8') as f:
            f.write("苏氏量化策略 - 优质股票筛选结果 (Scikit-learn + Optuna 神经网络评分 + Apriori 关联规则)\n")
            f.write(f"筛选时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型最终评分: {final_model_score:.4f}\n")  # 显示模型最终评分
            f.write(f"神经网络 R^2: {r2_score_nn:.4f}\n")  # 显示神经网络的 R^2
            f.write(f"关联规则数量: {len(rules)}\n")  # 显示关联规则数量
            f.write(f"最终筛选阈值: {threshold:.4f}\n")  # 显示神经网络的阈值
            f.write(f"优质股票数量: {len(quality_stocks)}\n")
            f.write("=" * 50 + "\n\n")

            for stock in quality_stocks:
                f.write(f"股票代码: {stock['代码']}\n")
                f.write(f"股票名称: {stock['名称']}\n")
                f.write(f"所属行业: {stock['行业']}\n")
                f.write(f"优质率: {stock['优质率']:.4f}\n")  # 显示神经网络的评分
                f.write(f"满足条件: {stock['满足条件']}\n")
                f.write(f"今日涨幅: {stock['涨幅']}\n")
                f.write("-" * 30 + "\n")

        print(f"\n✅ 优质股票已保存: {output_file2}")
    except Exception as e:
        print(f"\n❌ 无法保存优质股票: {e}")
        print(f"   错误信息: {e}")

    print(f"   找到 {len(quality_stocks)} 只优质股票（最终阈值={threshold:.4f}）")

    if len(quality_stocks) > 0:
        print(f"\n🎯 今日优质股票列表：")
        print("=" * 60)
        print(f"{'股票代码':<10} {'股票名称':<12} {'涨幅%':<8} {'优质率':<10}")
        print("-" * 60)
        for stock in quality_stocks[:12]: # 确保只打印前12个
            print(f"{stock['代码']:<10} {stock['名称']:<12} {stock['涨幅']:<8} {stock['优质率']:.4f}")
    else:
        print("\n⚠️ 今日没有找到符合条件的优质股票")
        print("   可能原因：")
        print("   1. 市场整体表现不佳，涨幅不足")
        print("   2. 数据获取不完整")
        print("3. 筛选条件过于严格，或模型区分度不足")

    print("\n" + "=" * 60)
    print("✅ 程序执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

