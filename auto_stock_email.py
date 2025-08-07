#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股自动化选股邮件发送系统
每天收盘后自动运行选股脚本并发送邮件通知
"""

import os
import sys
import smtplib
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
import time
import logging
import akshare as ak
from pathlib import Path

# 配置日志
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"stock_email_{datetime.now().strftime('%Y%m')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 清除代理设置
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['ALL_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

class StockEmailNotifier:
    """股票选股邮件通知系统"""
    
    def __init__(self):
        """初始化配置"""
        self.work_dir = Path(__file__).parent
        self.output_dir = self.work_dir / "输出数据"
        self.script_path = self.work_dir / "最终正确选股系统.py"
        
        # 加载邮件配置
        self.load_email_config()
        
    def load_email_config(self):
        """加载邮件配置（从环境变量或配置文件）"""
        # 首先尝试从环境变量读取
        self.sender_email = os.environ.get('STOCK_EMAIL_SENDER')
        self.sender_password = os.environ.get('STOCK_EMAIL_PASSWORD')
        self.receiver_emails = os.environ.get('STOCK_EMAIL_RECEIVERS', '').split(',')
        
        # 如果环境变量不存在，尝试从配置文件读取
        if not all([self.sender_email, self.sender_password, self.receiver_emails[0]]):
            config_file = Path.home() / ".stock_email_env"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    for line in f:
                        if line.startswith('#') or '=' not in line:
                            continue
                        key, value = line.strip().split('=', 1)
                        if key == 'STOCK_EMAIL_SENDER':
                            self.sender_email = value
                        elif key == 'STOCK_EMAIL_PASSWORD':
                            self.sender_password = value
                        elif key == 'STOCK_EMAIL_RECEIVERS':
                            # 支持多个收件人，用逗号分隔
                            self.receiver_emails = [email.strip() for email in value.split(',')]
                        elif key == 'STOCK_EMAIL_RECEIVER':
                            # 兼容旧配置
                            self.receiver_emails = [value.strip()]
            else:
                logger.error("邮件配置未找到！请设置环境变量或创建配置文件 ~/.stock_email_env")
                sys.exit(1)
        
        # 确保receiver_emails是列表
        if isinstance(self.receiver_emails, str):
            self.receiver_emails = [self.receiver_emails]
        
        # 过滤空邮箱
        self.receiver_emails = [e for e in self.receiver_emails if e and e.strip()]
    
    def is_trading_day(self):
        """判断今天是否是交易日"""
        today = datetime.now()
        
        # 周末不是交易日
        if today.weekday() >= 5:  # 0=周一, 6=周日
            logger.info(f"今天是周末，非交易日")
            return False
        
        # 使用akshare检查是否是交易日
        try:
            # 获取交易日历
            trade_dates = ak.tool_trade_date_hist_sina()
            today_str = today.strftime('%Y-%m-%d')
            
            # 检查今天是否在交易日列表中
            is_trading = today_str in trade_dates['trade_date'].values
            
            if not is_trading:
                logger.info(f"今天是节假日，非交易日")
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"无法获取交易日历，假设为交易日: {e}")
            # 如果无法获取交易日历，且不是周末，则假设是交易日
            return True
    
    def run_stock_selector(self):
        """运行选股脚本"""
        logger.info("开始运行选股脚本...")
        
        try:
            # 运行选股脚本
            result = subprocess.run(
                [sys.executable, str(self.script_path)],
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                logger.info("选股脚本执行成功")
                return True
            else:
                logger.error(f"选股脚本执行失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("选股脚本执行超时")
            return False
        except Exception as e:
            logger.error(f"运行选股脚本时出错: {e}")
            return False
    
    def read_stock_results(self):
        """读取选股结果"""
        result_file = self.output_dir / "优质股票.txt"
        
        if not result_file.exists():
            logger.error(f"选股结果文件不存在: {result_file}")
            return None
            
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"成功读取选股结果，文件大小: {len(content)} 字节")
            return content
        except Exception as e:
            logger.error(f"读取选股结果失败: {e}")
            return None
    
    def create_email_message(self, content, error_mode=False):
        """创建邮件消息"""
        msg = MIMEMultipart()
        
        # 设置邮件头
        if error_mode:
            msg['Subject'] = f"【错误】A股选股系统执行失败 - {datetime.now().strftime('%Y-%m-%d')}"
        else:
            msg['Subject'] = f"【A股选股】今日优质股票推荐 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        msg['From'] = self.sender_email
        msg['To'] = ', '.join(self.receiver_emails)  # 支持多个收件人
        
        # 创建邮件正文
        if error_mode:
            body = f"""
选股系统执行失败！

错误时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
错误信息：{content}

请检查系统日志：{log_file}
"""
        else:
            # 解析选股结果
            lines = content.split('\n')
            stock_count = 0
            for line in lines:
                if '优质股票数量:' in line:
                    stock_count = line.split(':')[1].strip()
                    break
            
            body = f"""
尊敬的投资者，您好！

今日A股选股系统已完成分析，为您筛选出 {stock_count} 只优质股票。

执行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
选股策略：苏氏量化策略
筛选阈值：2110

详细选股结果请查看附件。

===================
选股结果预览：
===================
{content[:1000]}...

【风险提示】
本选股结果仅供参考，不构成投资建议。股市有风险，投资需谨慎。

祝您投资顺利！

---
此邮件由A股自动化选股系统自动发送
"""
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 如果不是错误模式，添加附件
        if not error_mode:
            result_file = self.output_dir / "优质股票.txt"
            if result_file.exists():
                with open(result_file, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename="优质股票_{datetime.now().strftime("%Y%m%d")}.txt"'
                    )
                    msg.attach(part)
                    logger.info("已添加附件")
        
        return msg
    
    def send_email(self, msg, retry_times=3):
        """发送邮件（支持重试）"""
        for attempt in range(retry_times):
            try:
                logger.info(f"尝试发送邮件 (第{attempt + 1}次)...")
                
                # 连接到Gmail SMTP服务器
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()  # 启用TLS加密
                server.login(self.sender_email, self.sender_password)
                
                # 发送邮件给所有收件人
                text = msg.as_string()
                server.sendmail(self.sender_email, self.receiver_emails, text)
                server.quit()
                
                logger.info(f"邮件发送成功！收件人: {', '.join(self.receiver_emails)}")
                return True
                
            except Exception as e:
                logger.error(f"发送邮件失败 (第{attempt + 1}次): {e}")
                if attempt < retry_times - 1:
                    time.sleep(10)  # 等待10秒后重试
                    
        return False
    
    def run(self):
        """主运行函数"""
        logger.info("="*60)
        logger.info("A股自动化选股邮件系统启动")
        logger.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        # 检查是否是交易日
        if not self.is_trading_day():
            logger.info("今天不是交易日，跳过执行")
            return
        
        # 运行选股脚本
        if not self.run_stock_selector():
            # 发送错误通知
            msg = self.create_email_message("选股脚本执行失败", error_mode=True)
            self.send_email(msg)
            logger.error("选股脚本执行失败，已发送错误通知")
            return
        
        # 读取选股结果
        content = self.read_stock_results()
        if not content:
            # 发送错误通知
            msg = self.create_email_message("无法读取选股结果", error_mode=True)
            self.send_email(msg)
            logger.error("无法读取选股结果，已发送错误通知")
            return
        
        # 创建并发送邮件
        msg = self.create_email_message(content)
        if self.send_email(msg):
            logger.info("✅ 选股邮件发送成功！")
        else:
            logger.error("❌ 选股邮件发送失败！")
        
        logger.info("="*60)
        logger.info("A股自动化选股邮件系统执行完成")
        logger.info("="*60)


def main():
    """主函数"""
    notifier = StockEmailNotifier()
    notifier.run()


if __name__ == "__main__":
    main()