#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速选股 - 极简版
一行命令搞定选股和发邮件
"""

import sys
import os

def main():
    """主函数"""
    print("="*50)
    print("A股快速选股系统")
    print("="*50)
    print()
    
    # 询问操作
    print("请选择操作：")
    print("1. 仅选股（不发邮件）")
    print("2. 选股并发送邮件")
    print("3. 仅发送上次的选股结果")
    print("0. 退出")
    print()
    
    choice = input("请输入选项 (1/2/3/0): ").strip()
    
    if choice == "0":
        print("退出程序")
        return
    
    elif choice == "1":
        print("\n正在运行选股系统...")
        os.system("python3 最终正确选股系统.py")
        print("\n✅ 选股完成！结果保存在 输出数据/ 文件夹")
        
    elif choice == "2":
        print("\n正在运行选股并发送邮件...")
        os.system("python3 auto_stock_email.py")
        print("\n✅ 完成！请查收邮件")
        
    elif choice == "3":
        print("\n正在发送邮件...")
        # 只发送邮件，不重新选股
        from pathlib import Path
        result_file = Path("输出数据/优质股票.txt")
        if result_file.exists():
            # 直接调用邮件发送功能
            from auto_stock_email import StockEmailNotifier
            notifier = StockEmailNotifier()
            content = notifier.read_stock_results()
            if content:
                msg = notifier.create_email_message(content)
                if notifier.send_email(msg):
                    print("\n✅ 邮件发送成功！")
                else:
                    print("\n❌ 邮件发送失败")
            else:
                print("\n❌ 无法读取选股结果")
        else:
            print("\n❌ 没有找到选股结果，请先运行选股")
    
    else:
        print("无效选项")
    
    print()
    input("按回车键退出...")

if __name__ == "__main__":
    main()