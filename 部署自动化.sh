#!/bin/bash

# A股自动化选股邮件系统 - 快速部署脚本

echo "=========================================="
echo "A股自动化选股邮件系统 - 一键部署"
echo "=========================================="

# 1. 创建邮件配置文件
if [ ! -f "$HOME/.stock_email_env" ]; then
    echo ""
    echo "========== 快速配置邮件 =========="
    echo ""
    echo "请输入以下信息："
    echo ""
    
    # 交互式输入
    read -p "1. 你的Gmail邮箱: " gmail_sender
    echo "   (例如: mystock@gmail.com)"
    echo ""
    
    read -p "2. Gmail应用密码(16位): " gmail_password
    echo "   (获取地址: https://myaccount.google.com/security)"
    echo ""
    
    read -p "3. 接收邮箱(多个用逗号分隔): " receivers
    echo "   (例如: 123456@qq.com,backup@163.com)"
    echo ""
    
    # 创建配置文件
    cat > "$HOME/.stock_email_env" << EOF
# Gmail发件箱
STOCK_EMAIL_SENDER=$gmail_sender

# Gmail应用密码
STOCK_EMAIL_PASSWORD=$gmail_password

# 收件邮箱（支持QQ邮箱等任何邮箱）
STOCK_EMAIL_RECEIVERS=$receivers
EOF
    
    echo "✅ 配置已保存！"
    echo ""
fi

# 2. 安装Python依赖
echo "检查Python依赖..."
pip3 install akshare pandas numpy -q

# 3. 创建日志目录
mkdir -p logs

# 4. 创建launchd配置
PLIST_FILE="$HOME/Library/LaunchAgents/com.stock.daily.plist"
cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.stock.daily</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(which python3)</string>
        <string>$(pwd)/auto_stock_email.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$(pwd)</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>15</integer>
        <key>Minute</key>
        <integer>30</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$(pwd)/logs/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$(pwd)/logs/stderr.log</string>
</dict>
</plist>
EOF

# 5. 加载定时任务
launchctl unload "$PLIST_FILE" 2>/dev/null
launchctl load "$PLIST_FILE"

echo ""
echo "✅ 部署完成！"
echo ""
echo "测试命令："
echo "  python3 auto_stock_email.py"
echo ""
echo "服务将在每个交易日15:30自动运行"