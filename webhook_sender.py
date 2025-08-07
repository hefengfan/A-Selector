#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过Webhook发送选股结果
支持：企业微信、钉钉、飞书
"""

import requests
import json
from datetime import datetime
from pathlib import Path
import os

def send_to_wechat_work(webhook_url, content):
    """发送到企业微信"""
    data = {
        "msgtype": "text",
        "text": {
            "content": content
        }
    }
    response = requests.post(webhook_url, json=data)
    return response.json()

def send_to_dingtalk(webhook_url, content):
    """发送到钉钉"""
    data = {
        "msgtype": "text",
        "text": {
            "content": content
        }
    }
    response = requests.post(webhook_url, json=data)
    return response.json()

def send_to_feishu(webhook_url, content):
    """发送到飞书"""
    data = {
        "msg_type": "text",
        "content": {
            "text": content
        }
    }
    response = requests.post(webhook_url, json=data)
    return response.json()

def main():
    """主函数"""
    # 读取选股结果
    result_file = Path("输出数据/优质股票.txt")
    if not result_file.exists():
        print("选股结果文件不存在")
        return
    
    with open(result_file, 'r', encoding='utf-8') as f:
        stock_content = f.read()
    
    # 构建消息内容
    message = f"""【A股选股通知】
日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}
策略：苏氏量化策略

{stock_content}
"""
    
    # 从环境变量获取webhook地址
    webhook_url = os.environ.get('WEBHOOK_URL')
    webhook_type = os.environ.get('WEBHOOK_TYPE', 'wechat')  # wechat/dingtalk/feishu
    
    if not webhook_url:
        print("请设置WEBHOOK_URL环境变量")
        return
    
    # 发送消息
    if webhook_type == 'wechat':
        result = send_to_wechat_work(webhook_url, message)
    elif webhook_type == 'dingtalk':
        result = send_to_dingtalk(webhook_url, message)
    elif webhook_type == 'feishu':
        result = send_to_feishu(webhook_url, message)
    else:
        print(f"不支持的类型: {webhook_type}")
        return
    
    print(f"发送结果: {result}")

if __name__ == "__main__":
    main()