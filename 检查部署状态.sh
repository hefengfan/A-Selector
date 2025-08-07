#!/bin/bash

# 检查GitHub部署状态

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

clear

echo -e "${CYAN}================================="
echo -e "   GitHub部署状态检查"
echo -e "=================================${NC}"
echo ""

# 检查git状态
echo -e "${YELLOW}1. 检查Git配置${NC}"
if [ -d ".git" ]; then
    echo -e "${GREEN}✅ Git仓库已初始化${NC}"
    
    # 检查远程仓库
    if git remote -v | grep -q origin; then
        REMOTE_URL=$(git remote get-url origin)
        echo -e "${GREEN}✅ 远程仓库: $REMOTE_URL${NC}"
        
        # 提取用户名和仓库名
        if [[ $REMOTE_URL =~ github\.com[:/]([^/]+)/([^/.]+)(\.git)?$ ]]; then
            GITHUB_USER="${BASH_REMATCH[1]}"
            REPO_NAME="${BASH_REMATCH[2]}"
            echo -e "${GREEN}✅ 用户: $GITHUB_USER${NC}"
            echo -e "${GREEN}✅ 仓库: $REPO_NAME${NC}"
        fi
    else
        echo -e "${RED}❌ 未配置远程仓库${NC}"
        echo "  运行: bash GitHub一键部署.sh"
    fi
else
    echo -e "${RED}❌ Git仓库未初始化${NC}"
    echo "  运行: bash GitHub一键部署.sh"
fi

# 检查工作流文件
echo ""
echo -e "${YELLOW}2. 检查GitHub Actions配置${NC}"
if [ -f ".github/workflows/daily_stock.yml" ]; then
    echo -e "${GREEN}✅ 工作流文件已创建${NC}"
else
    echo -e "${RED}❌ 工作流文件不存在${NC}"
fi

# 检查必要文件
echo ""
echo -e "${YELLOW}3. 检查必要文件${NC}"
FILES=("最终正确选股系统.py" "auto_stock_email.py")
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ $file 缺失${NC}"
    fi
done

# 检查Gmail配置
echo ""
echo -e "${YELLOW}4. 检查本地邮件配置${NC}"
if [ -f "$HOME/.stock_email_env" ]; then
    echo -e "${GREEN}✅ 本地邮件配置已存在${NC}"
    # 检查配置内容
    if grep -q "your_email@gmail.com" "$HOME/.stock_email_env"; then
        echo -e "${YELLOW}⚠️  请编辑配置文件填入实际邮箱${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  本地邮件配置未创建（GitHub Actions不需要）${NC}"
fi

# 提供快速链接
if [ ! -z "$GITHUB_USER" ] && [ ! -z "$REPO_NAME" ]; then
    echo ""
    echo -e "${CYAN}================================="
    echo -e "   快速访问链接"
    echo -e "=================================${NC}"
    echo ""
    echo "仓库主页:"
    echo -e "${CYAN}https://github.com/$GITHUB_USER/$REPO_NAME${NC}"
    echo ""
    echo "Actions页面:"
    echo -e "${CYAN}https://github.com/$GITHUB_USER/$REPO_NAME/actions${NC}"
    echo ""
    echo "配置Secrets:"
    echo -e "${CYAN}https://github.com/$GITHUB_USER/$REPO_NAME/settings/secrets/actions${NC}"
    echo ""
    echo -e "${YELLOW}提示：按住Cmd点击链接可直接打开${NC}"
fi

echo ""
echo -e "${CYAN}=================================${NC}"
echo ""

# 提供下一步建议
if [ -d ".git" ] && git remote -v | grep -q origin; then
    echo -e "${GREEN}状态良好！${NC}"
    echo ""
    echo "下一步："
    echo "1. 确保已配置GitHub Secrets"
    echo "2. 访问Actions页面手动触发测试"
else
    echo -e "${YELLOW}需要部署${NC}"
    echo ""
    echo "运行: ${GREEN}bash GitHub一键部署.sh${NC}"
fi

echo ""