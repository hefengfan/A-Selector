#!/bin/bash

# GitHub一键部署脚本 - 自动化全流程
# 作者：A股自动化选股系统

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 清屏
clear

echo -e "${CYAN}================================================"
echo -e "     GitHub 一键部署 - A股自动化选股系统"
echo -e "================================================${NC}"
echo ""
echo -e "${GREEN}这个脚本将帮你：${NC}"
echo "  1. 创建GitHub仓库"
echo "  2. 上传所有代码"
echo "  3. 配置邮箱密码"
echo "  4. 启用自动化"
echo ""
echo -e "${YELLOW}准备工作：${NC}"
echo "  • 确保有GitHub账号"
echo "  • 准备好Gmail应用密码"
echo ""
read -p "准备好了吗？按回车继续..."

# 检查git是否安装
echo ""
echo -e "${BLUE}[步骤1/7] 检查环境...${NC}"
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git未安装！${NC}"
    echo "请先安装Git："
    echo "  brew install git"
    exit 1
fi
echo -e "${GREEN}✅ Git已安装${NC}"

# 检查是否已有.git目录
if [ -d ".git" ]; then
    echo -e "${YELLOW}⚠️  检测到已有Git仓库${NC}"
    read -p "是否删除旧仓库重新初始化？(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .git
        echo -e "${GREEN}✅ 已清理旧仓库${NC}"
    fi
fi

# 获取GitHub用户信息
echo ""
echo -e "${BLUE}[步骤2/7] GitHub账号配置${NC}"
echo ""

# 检查是否已有GitHub用户名
GITHUB_USER=$(git config --global user.name 2>/dev/null || echo "")
if [ -z "$GITHUB_USER" ]; then
    read -p "请输入你的GitHub用户名: " GITHUB_USER
    git config --global user.name "$GITHUB_USER"
else
    echo -e "检测到GitHub用户名: ${GREEN}$GITHUB_USER${NC}"
    read -p "是否使用此用户名？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        read -p "请输入新的GitHub用户名: " GITHUB_USER
        git config --global user.name "$GITHUB_USER"
    fi
fi

# 检查是否已有GitHub邮箱
GITHUB_EMAIL=$(git config --global user.email 2>/dev/null || echo "")
if [ -z "$GITHUB_EMAIL" ]; then
    read -p "请输入你的GitHub邮箱: " GITHUB_EMAIL
    git config --global user.email "$GITHUB_EMAIL"
else
    echo -e "检测到GitHub邮箱: ${GREEN}$GITHUB_EMAIL${NC}"
fi

# 仓库名称
echo ""
echo -e "${BLUE}[步骤3/7] 设置仓库名称${NC}"
REPO_NAME="stock-selector"
echo -e "建议仓库名: ${GREEN}$REPO_NAME${NC}"
read -p "使用默认名称？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    read -p "请输入仓库名称(只能包含字母、数字、-): " REPO_NAME
fi

# 初始化Git仓库
echo ""
echo -e "${BLUE}[步骤4/7] 初始化Git仓库...${NC}"
git init
git add .
git commit -m "初始化A股选股系统" || true
echo -e "${GREEN}✅ 本地仓库已创建${NC}"

# 创建GitHub仓库
echo ""
echo -e "${BLUE}[步骤5/7] 创建GitHub远程仓库${NC}"
echo ""
echo -e "${YELLOW}请按以下步骤操作：${NC}"
echo ""
echo "1. 打开浏览器，访问: ${CYAN}https://github.com/new${NC}"
echo ""
echo "2. 填写仓库信息："
echo "   • Repository name: ${GREEN}$REPO_NAME${NC}"
echo "   • Description: A股自动化选股系统"
echo "   • 选择: ${GREEN}Private${NC} (私有仓库)"
echo "   • 不要勾选任何初始化选项"
echo ""
echo "3. 点击 ${GREEN}Create repository${NC}"
echo ""
echo "4. 创建成功后，GitHub会显示仓库地址，类似："
echo "   ${CYAN}https://github.com/$GITHUB_USER/$REPO_NAME.git${NC}"
echo ""
read -p "完成后按回车继续..."

# 连接远程仓库
echo ""
echo -e "${BLUE}[步骤6/7] 连接远程仓库${NC}"
REPO_URL="https://github.com/$GITHUB_USER/$REPO_NAME.git"
echo -e "仓库地址: ${CYAN}$REPO_URL${NC}"
read -p "确认地址正确？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    read -p "请输入正确的仓库地址: " REPO_URL
fi

# 添加远程仓库
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"

# 推送代码
echo ""
echo -e "${BLUE}正在推送代码到GitHub...${NC}"
echo -e "${YELLOW}首次推送需要输入GitHub用户名和密码/token${NC}"
echo ""
echo "注意："
echo "• 用户名：你的GitHub用户名"
echo "• 密码：你的GitHub Personal Access Token"
echo "  (不是登录密码！获取Token: https://github.com/settings/tokens)"
echo ""

git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 代码推送成功！${NC}"
else
    echo -e "${RED}❌ 推送失败，请检查用户名和Token${NC}"
    echo ""
    echo "获取Token的方法："
    echo "1. 访问 https://github.com/settings/tokens"
    echo "2. 点击 Generate new token (classic)"
    echo "3. 勾选 repo 权限"
    echo "4. 生成并复制Token"
    exit 1
fi

# 配置Secrets
echo ""
echo -e "${BLUE}[步骤7/7] 配置邮件密码（GitHub Secrets）${NC}"
echo ""
echo -e "${YELLOW}最后一步：配置邮箱密码${NC}"
echo ""
echo "请按以下步骤操作："
echo ""
echo "1. 打开: ${CYAN}https://github.com/$GITHUB_USER/$REPO_NAME/settings/secrets/actions${NC}"
echo ""
echo "2. 点击 ${GREEN}New repository secret${NC} 添加以下3个密钥："
echo ""
echo "   第1个密钥："
echo "   • Name: ${GREEN}GMAIL_SENDER${NC}"
echo "   • Value: 你的Gmail邮箱 (如: mystock@gmail.com)"
echo ""
echo "   第2个密钥："
echo "   • Name: ${GREEN}GMAIL_PASSWORD${NC}"
echo "   • Value: Gmail应用密码 (16位，如: abcdefghijklmnop)"
echo ""
echo "   第3个密钥："
echo "   • Name: ${GREEN}EMAIL_RECEIVERS${NC}"
echo "   • Value: 收件邮箱 (如: 123456@qq.com,789@163.com)"
echo ""
read -p "配置完成后按回车继续..."

# 启用Actions
echo ""
echo -e "${CYAN}================================================${NC}"
echo -e "${GREEN}🎉 恭喜！部署即将完成！${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""
echo "最后一步：启用GitHub Actions"
echo ""
echo "1. 打开: ${CYAN}https://github.com/$GITHUB_USER/$REPO_NAME/actions${NC}"
echo ""
echo "2. 如果看到提示，点击: ${GREEN}I understand my workflows, go ahead and enable them${NC}"
echo ""
echo "3. 找到 ${GREEN}每日A股选股${NC} 工作流"
echo ""
echo "4. 点击 ${GREEN}Run workflow${NC} 按钮测试"
echo ""
echo -e "${CYAN}================================================${NC}"
echo -e "${GREEN}✅ 部署完成！${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""
echo "📋 重要信息："
echo "• 仓库地址: ${CYAN}https://github.com/$GITHUB_USER/$REPO_NAME${NC}"
echo "• 手动触发: ${CYAN}https://github.com/$GITHUB_USER/$REPO_NAME/actions${NC}"
echo "• 自动执行: 每个交易日15:30"
echo ""
echo "📱 手机访问："
echo "可以用手机浏览器打开Actions页面随时触发"
echo ""
echo -e "${GREEN}祝你投资顺利！${NC}"