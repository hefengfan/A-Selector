#!/bin/bash

# 测试GitHub Actions部署

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

clear

echo -e "${CYAN}================================="
echo -e "   GitHub Actions 测试工具"
echo -e "=================================${NC}"
echo ""

# 检查git配置
if [ -d ".git" ] && git remote -v | grep -q origin; then
    REMOTE_URL=$(git remote get-url origin)
    
    # 提取用户名和仓库名
    if [[ $REMOTE_URL =~ github\.com[:/]([^/]+)/([^/.]+)(\.git)?$ ]]; then
        GITHUB_USER="${BASH_REMATCH[1]}"
        REPO_NAME="${BASH_REMATCH[2]}"
        
        echo -e "${GREEN}✅ 检测到GitHub仓库${NC}"
        echo "   用户: $GITHUB_USER"
        echo "   仓库: $REPO_NAME"
        echo ""
        
        # 提供手动触发链接
        echo -e "${YELLOW}方法1：浏览器手动触发（推荐）${NC}"
        echo "========================================="
        echo ""
        echo "1. 打开下面的链接："
        echo -e "${CYAN}https://github.com/$GITHUB_USER/$REPO_NAME/actions${NC}"
        echo ""
        echo "2. 点击左侧 '每日A股选股'"
        echo ""
        echo "3. 点击右上角 'Run workflow' 按钮"
        echo ""
        echo "4. 再点击绿色 'Run workflow' 确认"
        echo ""
        echo "5. 等待1-2分钟，查看邮箱"
        echo ""
        echo -e "${YELLOW}提示：${NC}按住Cmd点击链接可直接打开"
        echo ""
        
        # 查看最近运行
        echo "========================================="
        echo -e "${YELLOW}查看运行状态${NC}"
        echo "========================================="
        echo ""
        echo "状态页面："
        echo -e "${CYAN}https://github.com/$GITHUB_USER/$REPO_NAME/actions${NC}"
        echo ""
        echo "• 🟡 黄色圆圈 = 运行中"
        echo "• ✅ 绿色勾 = 成功"  
        echo "• ❌ 红色叉 = 失败"
        echo ""
        
        # 故障排查
        echo "========================================="
        echo -e "${YELLOW}没收到邮件？检查以下项目${NC}"
        echo "========================================="
        echo ""
        echo "1. 检查垃圾邮件箱"
        echo ""
        echo "2. 确认Secrets配置："
        echo -e "   ${CYAN}https://github.com/$GITHUB_USER/$REPO_NAME/settings/secrets/actions${NC}"
        echo "   需要配置："
        echo "   • GMAIL_SENDER (发件邮箱)"
        echo "   • GMAIL_PASSWORD (应用密码)"
        echo "   • EMAIL_RECEIVERS (收件邮箱)"
        echo ""
        echo "3. 查看运行日志："
        echo "   点击失败的运行记录查看详细错误"
        echo ""
        
    else
        echo -e "${RED}❌ 无法解析仓库信息${NC}"
    fi
else
    echo -e "${RED}❌ 未检测到Git仓库${NC}"
    echo "请先运行: bash GitHub一键部署.sh"
fi

echo "========================================="
echo ""
read -p "按回车键退出..."