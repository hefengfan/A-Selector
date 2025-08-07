#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用QQ邮箱发送选股结果
QQ邮箱 → QQ邮箱，更简单！
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path
import os

class QQEmailSender:
    """QQ邮箱发送器"""
    
    def __init__(self):
        # QQ邮箱配置
        self.smtp_server = "smtp.qq.com"
        self.smtp_port = 587  # 或 465 (SSL)
        
        # 从环境变量读取
        self.sender = os.environ.get('QQ_EMAIL_SENDER')  # 你的QQ邮箱
        self.password = os.environ.get('QQ_EMAIL_PASSWORD')  # QQ邮箱授权码（不是QQ密码）
        self.receiver = os.environ.get('QQ_EMAIL_RECEIVER')  # 接收邮箱
    
    def send_stock_result(self):
        """发送选股结果"""
        # 读取选股结果
        result_file = Path("输出数据/优质股票.txt")
        if not result_file.exists():
            print("选股结果文件不存在")
            return False
        
        with open(result_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建邮件
        msg = MIMEMultipart()
        msg['Subject'] = f"【A股选股】{datetime.now().strftime('%Y-%m-%d')} 优质股票"
        msg['From'] = self.sender
        msg['To'] = self.receiver
        
        # 邮件正文
        body = f"""
今日A股选股结果

时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}
策略：苏氏量化策略

==================
{content}
==================

此邮件由自动化系统发送
"""
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 添加附件
        with open(result_file, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename="stocks_{datetime.now().strftime("%Y%m%d")}.txt"'
            )
            msg.attach(part)
        
        # 发送邮件
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender, self.password)
            server.send_message(msg)
            server.quit()
            print("✅ 邮件发送成功")
            return True
        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False

if __name__ == "__main__":
    sender = QQEmailSender()
    sender.send_stock_result()