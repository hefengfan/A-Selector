# A股自动化选股系统

## 系统功能
每天自动运行选股系统，筛选优质A股股票，并通过QQ邮箱发送通知。

## 核心文件（仅保留以下文件）

### 1. 最终正确选股系统.py
- 主要选股逻辑
- 基于苏氏量化策略筛选12只优质股票
- 输出到 `输出数据/优质股票.txt`

### 2. qq_stock_email.py  
- QQ邮箱发送模块
- 读取选股结果并发送邮件通知
- 支持多个收件人

### 3. .github/workflows/qq_email_stock.yml
- GitHub Actions工作流配置
- 每天15:30自动运行（北京时间）
- 自动执行选股并发送邮件

## 使用说明

### 配置GitHub Secrets（必须）
在GitHub仓库设置中添加以下Secrets：
- `QQ_EMAIL_SENDER`: rofanyou@qq.com
- `QQ_EMAIL_PASSWORD`: wzvecdxffpeqbdig
- `QQ_EMAIL_RECEIVERS`: rofanyou@qq.com

### 手动运行
```bash
# 本地运行选股
python 最终正确选股系统.py

# 发送邮件（需设置环境变量）
python qq_stock_email.py
```

### GitHub Actions自动运行
- 每天北京时间15:30自动执行
- 可在Actions页面手动触发
- 运行结果会通过邮件发送

## 输出文件
- `输出数据/优质股票.txt` - 选股结果
- `输出数据/A股数据.csv` - 完整数据
- `logs/` - 运行日志

## 注意事项
- QQ邮箱需要开启SMTP服务
- GitHub Secrets必须正确配置
- 选股仅供参考，不构成投资建议