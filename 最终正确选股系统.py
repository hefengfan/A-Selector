#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态选股系统 - 根据每天实时数据筛选
基于苏氏量化策略的真实计算逻辑
集成神经网络进行精准评分 (TensorFlow + Optuna)
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 导入 TensorFlow 和 Optuna 相关库
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import optuna

# 清除代理设置
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['ALL_PROXY'] = ''
os.environ['NO_PROXY'] = '*'


def calculate_features(row):
    """
    根据苏氏量化策略计算特征值，用于神经网络训练
    """
    features = []

    # F列：价格位置条件
    try:
        low_str = str(row['最低']).strip()
        ma60_str = str(row['60日均价']).strip()
        ma20_str = str(row['20日均价']).strip()
        current_str = str(row['最新']).strip()

        if '--' not in low_str and '--' not in ma60_str:
            low = float(low_str)
            ma60 = float(ma60_str)
            current = float(current_str)
            ma20 = float(ma20_str) if '--' not in ma20_str else 0

            condition_met = False
            if ma60 > 0 and 0.85 <= low / ma60 <= 1.15:
                condition_met = True
            if not condition_met and ma20 > 0 and 0.90 <= current / ma20 <= 1.10:
                condition_met = True

            features.append(1 if condition_met else 0)  # 1 or 0
        else:
            features.append(0)
    except:
        features.append(0)

    # G列：涨幅和价格位置
    try:
        change_str = str(row['涨幅%']).strip()
        current_str = str(row['最新']).strip()
        high_str = str(row['最高']).strip()
        low_str = str(row['最低']).strip()

        if '--' not in change_str:
            change = float(change_str)
            current = float(current_str)
            high = float(high_str)
            low = float(low_str)

            if change >= 5.0:
                threshold = high - (high - low) * 0.30
                features.append(1 if current >= threshold else 0)
            else:
                features.append(0)
        else:
            features.append(0)
    except:
        features.append(0)

    # H列：净利润>=3000万
    try:
        profit_str = str(row['归属净利润']).strip()
        profit = 0

        if '亿' in profit_str:
            profit = float(profit_str.replace('亿', ''))
        elif '万' in profit_str:
            profit = float(profit_str.replace('万', '')) / 10000

        features.append(profit)  # 直接使用净利润数值
    except:
        features.append(0)

    # I列：换手率<=20%
    try:
        turnover_str = str(row['实际换手%']).strip()
        if '--' not in turnover_str:
            turnover = float(turnover_str)
            features.append(turnover)  # 直接使用换手率数值
        else:
            features.append(100) # 换手率缺失时，赋予一个较大的值
    except:
        features.append(100)

    # J列：市值>=300亿
    try:
        cap_str = str(row['总市值']).strip()
        cap = 0

        if '万亿' in cap_str:
            cap = float(cap_str.replace('万亿', '')) * 10000
        elif '亿' in cap_str:
            cap = float(cap_str.replace('亿', ''))

        features.append(cap)  # 直接使用市值数值
    except:
        features.append(0)

    return features


def create_model(trial, input_shape):
    """
    使用 Optuna 建议的超参数创建 TensorFlow 神经网络模型
    """
    n_layers = trial.suggest_int('n_layers', 1, 3)  # 建议层数
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    for i in range(n_layers):
        num_units = trial.suggest_int(f'n_units_{i}', 32, 256)  # 建议神经元数量
        activation = trial.suggest_categorical(f'activation_{i}', ['relu', 'tanh', 'sigmoid'])  # 建议激活函数
        model.add(tf.keras.layers.Dense(num_units, activation=activation))
        dropout_rate = trial.suggest_float(f'dropout_{i}', 0.0, 0.5)  # 建议 Dropout 率
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1))  # 输出层
    return model


def objective(trial, X_train, y_train, X_test, y_test):
    """
    Optuna 优化的目标函数
    """
    model = create_model(trial, (X_train.shape[1],))
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])  # 建议优化器
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)  # 建议学习率

    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    # 添加 EarlyStopping 回调
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=0)

    _, mse = model.evaluate(X_test, y_test, verbose=0)
    return mse


def train_neural_network(df):
    """
    训练神经网络模型 (TensorFlow + Optuna)，预测股票评分
    """

    # 1. 准备训练数据
    print("\n   准备训练数据...")
    X = []
    y = []  # 目标变量：涨幅作为评分的依据
    for _, row in df.iterrows():
        features = calculate_features(row)
        X.append(features)

        # 使用涨幅作为目标变量，也可以考虑其他指标
        try:
            change_str = str(row['涨幅%']).strip()
            if '--' not in change_str:
                y.append(float(change_str))
            else:
                y.append(0)  # 缺失涨幅时，赋予0
        except:
            y.append(0)

    X = np.array(X)
    y = np.array(y)

    # 移除包含 NaN 或无穷大的行
    mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) == 0:
        print("   ❌ 没有有效的训练数据，无法训练神经网络。")
        return None, None

    # 2. 数据预处理
    print("   数据预处理...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3. 划分训练集和测试集
    print("   划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 使用 Optuna 优化超参数
    print("   使用 Optuna 优化超参数...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=10)  # 调整 trials 数量

    # 5. 使用最佳超参数创建模型
    print("   使用最佳超参数创建模型...")
    best_model = create_model(study.best_trial, (X_train.shape[1],))
    best_model.compile(optimizer=study.best_params['optimizer'], loss='mse', metrics=['mse'])

    # 6. 训练最佳模型
    print("   训练最佳模型...")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    best_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=0)

    # 7. 评估最佳模型
    print("   评估最佳模型...")
    y_pred = best_model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"   均方误差 (MSE): {mse:.4f}")
    print(f"   R^2 Score: {r2:.4f}")

    print("   最佳超参数:")
    print(study.best_params)

    return best_model, scaler, r2


def predict_score_with_nn(row, model, scaler):
    """
    使用训练好的 TensorFlow 神经网络模型预测股票评分
    """
    features = calculate_features(row)
    features = np.array(features).reshape(1, -1)  # 转换为二维数组

    # 检查是否有缺失值或无穷值
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        return 0  # 如果有，返回一个默认值

    features_scaled = scaler.transform(features)
    score = model.predict(features_scaled, verbose=0)[0][0]
    return score


def main():
    """主程序"""
    print("\n" + "="*60)
    print("动态选股系统 - 实时计算版")
    print("集成神经网络进行精准评分 (TensorFlow + Optuna)")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 创建输出目录
    os.makedirs('输出数据', exist_ok=True)

    # ========== 第一步：获取数据 ==========
    print("\n1. 获取A股数据...")

    # 先尝试获取实时数据
    try:
        print("   尝试获取实时数据...")
        df = ak.stock_zh_a_spot_em()
        print(f"   ✅ 成功获取 {len(df)} 只股票的实时数据")

        # 保存原始代码
        df['原始代码'] = df['代码'].copy()

        # 格式化代码
        df['代码'] = df['代码'].apply(lambda x: f'= "{str(x)}"')

        # 格式化数值列
        for col in ['最新价', '最高', '最低', '开盘', '昨收']:
            if col in df.columns:
                new_col = col.replace('价', '')
                df[new_col] = col.apply(
                    lambda x: f" {float(x):.2f}" if pd.notna(x) and str(x) not in ['--', '', None] else " --"
                )

        if '涨跌幅' in df.columns:
            df['涨幅%'] = df['涨跌幅'].apply(
                lambda x: f" {float(x):.2f}" if pd.notna(x) else " --"
            )

        if '换手率' in df.columns:
            df['实际换手%'] = df['换手率'].apply(
                lambda x: f" {float(x):.2f}" if pd.notna(x) else " --"
            )

        # 处理名称
        df['名称'] = df['名称'].apply(lambda x: f" {x}" if not str(x).startswith(' ') else x)

        # 设置默认值
        df['所属行业'] = '  其他'
        df['20日均价'] = ' --'
        df['60日均价'] = ' --'
        df['归属净利润'] = ' --'
        df['市盈率(动)'] = ' --'
        df['总市值'] = ' --'

    except Exception as e:
        print(f"   ❌ 实时获取失败: {e}")
        print("   使用参考数据作为备选...")

        # 使用参考数据
        try:
            df = pd.read_csv('参考数据/Table.xls', sep='\t', encoding='gbk', dtype=str)
            print(f"   ✅ 从参考文件加载了 {len(df)} 条数据")
            df['原始代码'] = df['代码'].str.replace('= "', '').str.replace('"', '')
        except Exception as e2:
            print(f"   ❌ 无法加载参考数据: {e2}")
            return

    # 尝试补充均线和财务数据
    try:
        ref_df = pd.read_csv('参考数据/Table.xls', sep='\t', encoding='gbk', dtype=str)
        ref_map = {}
        for _, row in ref_df.iterrows():
            code = str(row['代码']).replace('= "', '').replace('"', '')
            ref_map[code] = row.to_dict()

        # 合并参考数据
        for i, code in enumerate(df.get('原始代码', [])):
            if code in ref_map:
                ref = ref_map[code]
                # 补充缺失的数据
                for col in ['20日均价', '60日均价', '所属行业', '归属净利润', '总市值', '市盈率(动)']:
                    if col in ref:
                        df.loc[i, col] = ref[col]

        print(f"   ✅ 补充了 {len(ref_map)} 条参考数据")
    except:
        print("   ⚠️ 无法补充参考数据")

    # 添加序号
    df['序'] = range(1, len(df) + 1)
    df['Unnamed: 16'] = ''

    # 选择输出列
    output_columns = [
        '序', '代码', '名称', '最新', '涨幅%', '最高', '最低',
        '实际换手%', '所属行业', '20日均价', '60日均价',
        '市盈率(动)', '总市值', '归属净利润', '昨收', '开盘', 'Unnamed: 16'
    ]

    for col in output_columns:
        if col not in df.columns:
            df[col] = ' --' if col != 'Unnamed: 16' else ''

    final_df = df[output_columns]

    # 保存A股数据
    output_file1 = '输出数据/A股数据.csv'
    final_df.to_csv(output_file1, index=False, encoding='utf-8-sig')
    print(f"\n✅ A股数据已保存: {output_file1}")
    print(f"   共 {len(final_df)} 只股票")

    # ========== 第二步：训练神经网络 ==========
    print("\n2. 训练神经网络模型...")
    model, scaler, r2_score = train_neural_network(final_df)

    if model is None:
        print("   ❌ 神经网络训练失败，无法进行后续筛选。")
        return

    # ========== 第三步：动态筛选优质股票 ==========
    print("\n3. 动态筛选优质股票...")

    # 创建一个包含所有股票评分的列
    final_df['神经网络评分'] = final_df.apply(lambda row: predict_score_with_nn(row, model, scaler), axis=1)

    quality_stocks = []
    threshold = 0.0  # 调整阈值以获得更多结果

    # 统计
    stats = {'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0}

    for idx, row in final_df.iterrows():
        score = row['神经网络评分']  # 直接使用神经网络评分
        conditions = ""  # 神经网络评分不需要条件

        # 统计（原始评分方式的统计，如果只用神经网络，可以移除）
        features = calculate_features(row)
        if features[0] == 1: stats['F'] += 1
        if features[1] == 1: stats['G'] += 1
        if features[2] > 0.3: stats['H'] += 1
        if features[3] <= 25: stats['I'] += 1
        if features[4] >= 200: stats['J'] += 1

        # 判断是否达标
        if score >= threshold:
            code = str(row['代码']).replace('= "', '').replace('"', '')
            quality_stocks.append({
                '代码': code,
                '名称': str(row['名称']).strip(),
                '行业': str(row['所属行业']).strip(),
                '优质率': score,
                '满足条件': conditions,
                '涨幅': str(row['涨幅%']).strip()
            })

    # 打印统计
    total = len(final_df)
    if total > 0:
        print(f"\n   条件满足统计（共{total}只股票）：")
        print(f"   F列(价格位置): {stats['F']}只 ({stats['F']/total*100:.1f}%)")
        print(f"   G列(涨幅条件): {stats['G']}只 ({stats['G']/total*100:.1f}%)")
        print(f"   H列(净利润): {stats['H']}只 ({stats['H']/total*100:.1f}%)")
        print(f"   I列(换手率): {stats['I']}只 ({stats['I']/total*100:.1f}%)")
        print(f"   J列(市值): {stats['J']}只 ({stats['J']/total*100:.1f}%)")

    # 按优质率降序排序
    quality_stocks = sorted(quality_stocks, key=lambda x: (x['优质率'], x['代码']), reverse=True)

    # 如果结果太少，尝试降低阈值
    if len(quality_stocks) < 10:
        print(f"\n   ⚠️ 只找到{len(quality_stocks)}只股票，尝试降低阈值...")
        threshold = np.percentile([stock['优质率'] for stock in quality_stocks], 25) if quality_stocks else 0  # 使用25%分位数作为阈值
        quality_stocks = []

        for idx, row in final_df.iterrows():
            score = row['神经网络评分']
            if score >= threshold:
                code = str(row['代码']).replace('= "', '').replace('"', '')
                quality_stocks.append({
                    '代码': code,
                    '名称': str(row['名称']).strip(),
                    '行业': str(row['所属行业']).strip(),
                    '优质率': score,
                    '满足条件': "",  # 神经网络评分不需要条件
                    '涨幅': str(row['涨幅%']).strip()
                })

        quality_stocks = sorted(quality_stocks, key=lambda x: (x['优质率'], x['代码']), reverse=True)
        quality_stocks = quality_stocks[:12]  # 只取前12只

    # 保存优质股票
    output_file2 = '输出数据/优质股票.txt'
    with open(output_file2, 'w', encoding='utf-8') as f:
        f.write("苏氏量化策略 - 优质股票筛选结果 (TensorFlow + Optuna 神经网络评分)\n")
        f.write(f"筛选时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型准确率 (R^2): {r2_score:.4f}\n") # 显示模型准确率
        f.write(f"筛选阈值: {threshold:.4f}\n")  # 显示神经网络的阈值
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
    print(f"   找到 {len(quality_stocks)} 只优质股票（阈值={threshold:.4f}）")  # 显示神经网络的阈值

    if len(quality_stocks) > 0:
        print(f"\n🎯 今日优质股票列表：")
        print("=" * 60)
        print("股票代码    股票名称        涨幅%      优质率")
        print("-" * 60)
        for stock in quality_stocks[:12]:
            print(f"{stock['代码']:8}    {stock['名称']:12}    {stock['涨幅']:6}    {stock['优质率']:.4f}")  # 显示神经网络的评分
    else:
        print("\n⚠️ 今日没有找到符合条件的优质股票")
        print("   可能原因：")
        print("   1. 市场整体表现不佳，涨幅不足")
        print("   2. 数据获取不完整")
        print("   3. 筛选条件过于严格")

    # 将包含神经网络评分的 DataFrame 保存到 CSV
    output_file1 = '输出数据/A股数据.csv'
    final_df.to_csv(output_file1, index=False, encoding='utf-8-sig')
    print(f"\n✅ 包含神经网络评分的 A 股数据已保存: {output_file1}")

    print("\n" + "=" * 60)
    print("✅ 程序执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
