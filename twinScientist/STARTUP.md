# TwinScientist 启动指南

## 环境就绪 ✅

Python + pip + 所有核心依赖已安装完毕。

## 最后一步：配置 API Key

### 方法 1：复制 .env.example 为 .env（推荐）

在 PowerShell 中执行：
```powershell
cd "C:\Users\KevinShin\Desktop\TwinScientist--main3\twinScientist"
Copy-Item .env.example .env
notepad .env
```

然后将 `BAILIAN_API_KEY=sk-your-key-here` 中的占位符替换为真实密钥。

### 方法 2：设置环境变量

在 PowerShell 中临时设置：
```powershell
$env:BAILIAN_API_KEY = "sk-your-real-key"
```

或系统级别永久设置：
```powershell
[Environment]::SetEnvironmentVariable("BAILIAN_API_KEY", "sk-your-real-key", "User")
```

## 运行命令

### CLI 模式（单次运行）
```powershell
cd twinScientist
python main.py --question "高温环境对老年人心率变异性有何影响？" --domain "环境健康" --iterations 10
```

### 交互模式
```powershell
cd twinScientist
python main.py
```

### Web UI 模式
```powershell
cd twinScientist
python main.py --ui
```
→ 浏览器访问 http://127.0.0.1:7860

## 可选依赖（暂不安装也可运行）

以下包用于高级因果推断功能（当前代码使用 fallback/granger/ccm 核心逻辑时不需要）：
- `ccauchy`, `causalgraphicalmodels`, `statsmodels`, `pgmpy`, `bnlearn`, `dowhy`, `torch`

如需启用这些功能：
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install statsmodels ccauchy dowhy bnlearn pgmpy
```

## 常见问题

**Q: 提示找不到 python？**  
A: 确认 Python 已安装并加入 PATH，或在终端中运行 `python --version` 验证

**Q: 提示缺少模块？**  
A: 可能需要重启终端以刷新 PATH，或手动安装缺失的包

**Q: LLM 调用超时？**  
A: 检查网络连接，阿里云 API 可能有时延迟较高

## 测试验证

已验证整个 pipeline 从 ethics_check → literature_review → orchestrator → hypothesis_generation → ... → END 全部正常运行（除 LLM API 调用外）。
