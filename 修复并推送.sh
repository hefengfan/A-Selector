#!/bin/bash

# 修复GitHub Actions版本问题并推送

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

clear

echo -e "${CYAN}================================="
echo -e "   修复GitHub Actions并推送"
echo -e "=================================${NC}"
echo ""

echo -e "${YELLOW}问题诊断：${NC}"
echo "你遇到的错误是因为GitHub Actions使用了已弃用的v3版本"
echo "需要升级到v4/v5版本"
echo ""

# 检查是否有git仓库
if [ ! -d ".git" ]; then
    echo -e "${RED}❌ 当前目录不是git仓库${NC}"
    echo "请先运行: bash GitHub一键部署.sh"
    exit 1
fi

echo -e "${GREEN}[步骤1/3] 拉取最新代码${NC}"
git pull origin main 2>/dev/null || echo "本地已是最新"

echo ""
echo -e "${GREEN}[步骤2/3] 提交修复${NC}"
git add .github/workflows/daily_stock.yml
git commit -m "修复: 升级GitHub Actions到最新版本(v4/v5)" || echo "没有更改需要提交"

echo ""
echo -e "${GREEN}[步骤3/3] 推送到GitHub${NC}"
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ 修复成功！${NC}"
    echo ""
    echo -e "${CYAN}================================="
    echo -e "   下一步操作"
    echo -e "=================================${NC}"
    echo ""
    echo "1. 打开Actions页面重新测试："
    echo -e "   ${CYAN}https://github.com/RuofanYou/A-Selector/actions${NC}"
    echo ""
    echo "2. 点击 '每日A股选股'"
    echo "3. 点击 'Run workflow'"
    echo "4. 点击绿色 'Run workflow' 确认"
    echo ""
    echo -e "${GREEN}这次应该能成功了！${NC}"
else
    echo ""
    echo -e "${RED}❌ 推送失败${NC}"
    echo "可能需要输入GitHub用户名和密码/token"
fi

echo ""
read -p "按回车键退出..."