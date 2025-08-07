# 🎉 QQ邮箱版本已部署成功！

## ✅ 已完成的操作
1. ✅ 创建了QQ邮箱发送脚本 `qq_stock_email.py`
2. ✅ 创建了新的工作流 `qq_email_stock.yml`
3. ✅ 推送到GitHub仓库

## 🔑 最后一步：配置GitHub Secrets

### 立即配置（复制以下信息）

打开这个链接：
**https://github.com/RuofanYou/A-Selector/settings/secrets/actions**

点击 **"New repository secret"** 添加以下3个密钥：

#### 密钥1：QQ邮箱账号
- **Name**: `QQ_EMAIL_SENDER`
- **Value**: `rofanyou@qq.com`

#### 密钥2：QQ邮箱授权码
- **Name**: `QQ_EMAIL_PASSWORD`
- **Value**: `wzvecdxffpeqbdig`

#### 密钥3：接收邮箱（可以多个）
- **Name**: `QQ_EMAIL_RECEIVERS`
- **Value**: `rofanyou@qq.com`
  - 如果要发送给多个邮箱，用逗号分隔
  - 例如：`rofanyou@qq.com,backup@163.com`

## 🚀 测试运行

配置完Secrets后，立即测试：

1. 打开：**https://github.com/RuofanYou/A-Selector/actions**
2. 点击左侧 **"每日A股选股 - QQ邮箱版"**
3. 点击右上角 **"Run workflow"**
4. 点击绿色 **"Run workflow"** 确认
5. 等待1-2分钟
6. 检查QQ邮箱！

## 📧 邮件特点

- **发件人**：rofanyou@qq.com
- **收件人**：rofanyou@qq.com（或你配置的其他邮箱）
- **邮件标题**：【A股选股】今日优质股票推荐 - 日期
- **附件**：优质股票.txt
- **执行时间**：每天15:30自动执行

## ⚠️ 重要提醒

1. **授权码不是QQ密码**，是专门的SMTP授权码
2. **首次可能在垃圾箱**，记得查看
3. **可以添加白名单**，确保收到邮件

## 🎯 快速链接

- 配置Secrets：https://github.com/RuofanYou/A-Selector/settings/secrets/actions
- 手动触发：https://github.com/RuofanYou/A-Selector/actions
- 查看运行记录：https://github.com/RuofanYou/A-Selector/actions

---

**现在就去配置Secrets，然后测试吧！**