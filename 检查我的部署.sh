#!/bin/bash

# 专门为你的仓库定制的检查脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

clear

echo -e "${CYAN}================================="
echo -e "   你的GitHub仓库状态检查"
echo -e "=================================${NC}"
echo ""

# 你的仓库信息
GITHUB_USER="RuofanYou"
REPO_NAME="A-Selector"

echo -e "${GREEN}✅ 仓库信息${NC}"
echo "   用户: $GITHUB_USER"
echo "   仓库: $REPO_NAME"
echo ""

echo -e "${CYAN}================================="
echo -e "   快速访问链接"
echo -e "=================================${NC}"
echo ""

echo -e "${YELLOW}1. 手动触发测试：${NC}"
echo -e "${BLUE}https://github.com/$GITHUB_USER/$REPO_NAME/actions${NC}"
echo "   → 点击 '每日A股选股'"
echo "   → 点击 'Run workflow'"
echo ""

echo -e "${YELLOW}2. 查看运行记录：${NC}"
echo -e "${BLUE}https://github.com/$GITHUB_USER/$REPO_NAME/actions${NC}"
echo ""

echo -e "${YELLOW}3. 检查Secrets配置：${NC}"
echo -e "${BLUE}https://github.com/$GITHUB_USER/$REPO_NAME/settings/secrets/actions${NC}"
echo ""

echo -e "${CYAN}================================="
echo -e "   Secrets配置检查清单"
echo -e "=================================${NC}"
echo ""
echo "确保已配置以下3个密钥："
echo ""
echo "1. ${GREEN}GMAIL_SENDER${NC}"
echo "   示例: mystock2025@gmail.com"
echo ""
echo "2. ${GREEN}GMAIL_PASSWORD${NC}"
echo "   示例: abcdefghijklmnop (16位应用密码)"
echo "   获取: https://myaccount.google.com/apppasswords"
echo ""
echo "3. ${GREEN}EMAIL_RECEIVERS${NC}"
echo "   示例: 123456@qq.com,backup@163.com"
echo ""

echo -e "${CYAN}================================="
echo -e "   测试步骤"
echo -e "=================================${NC}"
echo ""
echo "1. 点击上面的Actions链接"
echo "2. 点击 '每日A股选股' 工作流"
echo "3. 点击 'Run workflow' 按钮"
echo "4. 再点击绿色 'Run workflow' 确认"
echo "5. 等待1-2分钟"
echo "6. 检查邮箱（包括垃圾邮件箱）"
echo ""

echo -e "${CYAN}================================="
echo -e "   故障排查"
echo -e "=================================${NC}"
echo ""
echo "如果没收到邮件："
echo ""
echo "1. ${YELLOW}检查垃圾邮件箱${NC}"
echo "   QQ邮箱经常把Gmail邮件放垃圾箱"
echo ""
echo "2. ${YELLOW}查看运行日志${NC}"
echo "   点击失败的运行记录"
echo "   查看红色错误信息"
echo ""
echo "3. ${YELLOW}常见错误${NC}"
echo "   • Authentication failed → 密码错误"
echo "   • Secret not found → Secrets未配置"
echo "   • Network error → 重试即可"
echo ""

echo -e "${GREEN}提示：按住Cmd键点击链接可直接打开${NC}"
echo ""
read -p "按回车键退出..."