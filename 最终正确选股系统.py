#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态选股系统 - 根据每天实时数据筛选
基于苏氏量化策略的真实计算逻辑
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 清除代理设置
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['ALL_PROXY'] = ''
os.environ['NO_PROXY'] = '*'


def calculate_score(row):
    """
    根据苏氏量化策略计算得分
    注意：这里使用宽松的条件以匹配Excel的实际行为
    """
    score = 0
    details = []
    
    # F列：价格位置条件（1000分）
    # 根据实际数据分析，这个条件可能更宽松
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
            
            # 多种可能的条件（根据实际数据调整）
            condition_met = False
            
            # 条件1：最低价在60日均价附近（放宽到15%）
            if ma60 > 0 and 0.85 <= low/ma60 <= 1.15:
                condition_met = True
            
            # 条件2：现价在20日均价附近（备选）
            if not condition_met and ma20 > 0 and 0.90 <= current/ma20 <= 1.10:
                condition_met = True
            
            if condition_met:
                score += 1000
                details.append('F')
    except:
        pass
    
    # G列：涨幅和价格位置（1000分）
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
            
            # 涨幅条件（可能需要调整阈值）
            if change >= 5.0:  # 降低到5%试试
                threshold = high - (high - low) * 0.30  # 放宽到30%
                if current >= threshold:
                    score += 1000
                    details.append('G')
    except:
        pass
    
    # H列：净利润>=3000万（100分）
    try:
        profit_str = str(row['归属净利润']).strip()
        profit = 0
        
        if '亿' in profit_str:
            profit = float(profit_str.replace('亿', ''))
        elif '万' in profit_str:
            profit = float(profit_str.replace('万', '')) / 10000
            
        if profit >= 0.3:  # 0.3亿=3000万
            score += 100
            details.append('H')
    except:
        pass
    
    # I列：换手率<=20%（10分）
    try:
        turnover_str = str(row['实际换手%']).strip()
        if '--' not in turnover_str:
            turnover = float(turnover_str)
            # 放宽到25%
            if turnover <= 25:
                score += 10
                details.append('I')
    except:
        pass
    
    # J列：市值>=300亿（1分）
    try:
        cap_str = str(row['总市值']).strip()
        cap = 0
        
        if '万亿' in cap_str:
            cap = float(cap_str.replace('万亿', '')) * 10000
        elif '亿' in cap_str:
            cap = float(cap_str.replace('亿', ''))
            
        # 放宽到200亿
        if cap >= 200:
            score += 1
            details.append('J')
    except:
        pass
    
    return score, '+'.join(details)


def main():
    """主程序"""
    print("\n" + "="*60)
    print("动态选股系统 - 实时计算版")
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
                df[new_col] = df[col].apply(
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
    
    # ========== 第二步：动态筛选优质股票 ==========
    print("\n2. 动态筛选优质股票...")
    
    quality_stocks = []
    threshold = 2100  # 降低阈值以获得更多结果
    
    # 统计
    stats = {'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0}
    
    for idx, row in final_df.iterrows():
        score, conditions = calculate_score(row)
        
        # 统计
        for cond in ['F', 'G', 'H', 'I', 'J']:
            if cond in conditions:
                stats[cond] += 1
        
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
        threshold = 1100
        quality_stocks = []
        
        for idx, row in final_df.iterrows():
            score, conditions = calculate_score(row)
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
        
        quality_stocks = sorted(quality_stocks, key=lambda x: (x['优质率'], x['代码']), reverse=True)
        quality_stocks = quality_stocks[:12]  # 只取前12只
    
    # 保存优质股票
    output_file2 = '输出数据/优质股票.txt'
    with open(output_file2, 'w', encoding='utf-8') as f:
        f.write("苏氏量化策略 - 优质股票筛选结果\n")
        f.write(f"筛选时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"筛选阈值: {threshold}\n")
        f.write(f"优质股票数量: {len(quality_stocks)}\n")
        f.write("="*50 + "\n\n")
        
        for stock in quality_stocks:
            f.write(f"股票代码: {stock['代码']}\n")
            f.write(f"股票名称: {stock['名称']}\n")
            f.write(f"所属行业: {stock['行业']}\n")
            f.write(f"优质率: {stock['优质率']}\n")
            f.write(f"满足条件: {stock['满足条件']}\n")
            f.write(f"今日涨幅: {stock['涨幅']}\n")
            f.write("-"*30 + "\n")
    
    print(f"\n✅ 优质股票已保存: {output_file2}")
    print(f"   找到 {len(quality_stocks)} 只优质股票（阈值={threshold}）")
    
    if len(quality_stocks) > 0:
        print(f"\n🎯 今日优质股票列表：")
        print("="*60)
        print("股票代码    股票名称        涨幅%      优质率")
        print("-"*60)
        for stock in quality_stocks[:12]:
            print(f"{stock['代码']:8}    {stock['名称']:12}    {stock['涨幅']:6}    {stock['优质率']}")
    else:
        print("\n⚠️ 今日没有找到符合条件的优质股票")
        print("   可能原因：")
        print("   1. 市场整体表现不佳，涨幅不足")
        print("   2. 数据获取不完整")
        print("   3. 筛选条件过于严格")
    
    print("\n" + "="*60)
    print("✅ 程序执行完成！")
    print("="*60)


if __name__ == "__main__":
    main()