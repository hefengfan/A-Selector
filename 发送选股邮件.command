#!/bin/bash

# A股选股 - 一键发送邮件
# 双击此文件即可运行

# 进入脚本所在目录
cd "$(dirname "$0")"

# 清屏
clear

echo "======================================"
echo "     A股选股系统 - 一键发送"
echo "======================================"
echo ""
echo "正在执行选股并发送邮件..."
echo ""

# 检查配置文件
if [ ! -f "$HOME/.stock_email_env" ]; then
    echo "❌ 邮件配置未找到"
    echo ""
    echo "请先运行: bash 部署自动化.sh"
    echo ""
    read -p "按回车键退出..."
    exit 1
fi

# 运行选股并发送邮件
python3 auto_stock_email.py

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 选股邮件已发送成功！"
    echo ""
    echo "请查收邮箱"
else
    echo ""
    echo "❌ 执行失败，请查看错误信息"
fi

echo ""
echo "======================================"
read -p "按回车键关闭窗口..."