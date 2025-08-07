# 🚀 GitHub自动化部署 - 永远不需要开电脑！

## 为什么选择GitHub Actions？
- ✅ **完全免费**
- ✅ **不需要电脑开机**
- ✅ **GitHub服务器自动运行**
- ✅ **每月2000分钟免费额度（足够用一辈子）**

---

## 📝 10分钟部署步骤

### 第1步：上传代码到GitHub

1. 创建GitHub账号（如果没有）：https://github.com
2. 新建仓库（Private私有仓库即可）
3. 上传本项目所有文件

```bash
git init
git add .
git commit -m "初始化选股系统"
git remote add origin https://github.com/你的用户名/stock-selector.git
git push -u origin main
```

### 第2步：配置邮箱密码（安全存储）

1. 进入你的GitHub仓库
2. 点击 `Settings`（设置）
3. 左侧找到 `Secrets and variables` > `Actions`
4. 点击 `New repository secret` 添加以下3个密钥：

| 密钥名称 | 密钥值 | 说明 |
|---------|--------|------|
| `GMAIL_SENDER` | mystock@gmail.com | 你的Gmail邮箱 |
| `GMAIL_PASSWORD` | abcdefghijklmnop | Gmail应用密码（16位） |
| `EMAIL_RECEIVERS` | 123456@qq.com,789@163.com | 收件邮箱（多个逗号分隔） |

### 第3步：启用Actions

1. 点击仓库的 `Actions` 标签
2. 点击 `I understand my workflows, go ahead and enable them`
3. 找到 `每日A股选股` 工作流
4. 完成！

---

## 🎯 使用方法

### 自动执行
- **执行时间**：每个交易日北京时间15:30
- **自动判断**：周末和节假日不执行
- **邮件通知**：自动发送到你的邮箱

### 手动触发
1. 进入 `Actions` 标签
2. 选择 `每日A股选股`
3. 点击 `Run workflow`
4. 点击绿色 `Run workflow` 按钮
5. 等待1-2分钟即可收到邮件

### 查看历史记录
- 在 `Actions` 标签可以看到所有执行记录
- 点击任意记录可以下载选股结果文件

---

## 🔧 高级设置

### 修改执行时间
编辑 `.github/workflows/daily_stock.yml` 文件：
```yaml
schedule:
  - cron: '30 7 * * 1-5'  # UTC时间，北京时间-8小时
```

时间对照表：
- 北京时间 09:30 → UTC 01:30 → cron: '30 1 * * 1-5'
- 北京时间 15:30 → UTC 07:30 → cron: '30 7 * * 1-5'
- 北京时间 20:00 → UTC 12:00 → cron: '0 12 * * 1-5'

### 添加更多收件人
在 `EMAIL_RECEIVERS` 密钥中添加，用逗号分隔：
```
123@qq.com,456@163.com,789@gmail.com
```

---

## ❓ 常见问题

**Q: GitHub Actions安全吗？**
A: 非常安全！密码存储在GitHub Secrets中，加密保存，即使是你自己也看不到。

**Q: 会不会超出免费额度？**
A: 不会！每次运行约2分钟，每月60分钟，免费额度2000分钟，用不完。

**Q: 可以在手机上触发吗？**
A: 可以！用手机浏览器登录GitHub，进入Actions手动触发。

**Q: 邮件没收到怎么办？**
A: 检查垃圾邮件箱，或查看Actions运行日志排查问题。

---

## 🎉 恭喜！

设置完成后，你就拥有了一个：
- 永远不需要开电脑
- 完全自动化
- 免费稳定
- 支持手机查看

的A股选股系统！

---

## 📱 额外福利：手机快捷方式

iOS用户可以创建Safari书签，直接打开：
```
https://github.com/你的用户名/stock-selector/actions
```

一键查看最新选股结果！