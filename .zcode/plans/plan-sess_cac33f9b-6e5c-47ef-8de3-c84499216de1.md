## 目标
把嵌套的 `twinScientist/` 子目录内容**向上提升一级**到 git 仓库根目录 `TwinScientist/`，使项目直接位于仓库根（消除嵌套）。然后修复因此断裂的 3 处代码引用、修整文档路径，提交并推送到 origin/main。

> 说明：仓库根目录名是 `TwinScientist`（大写），项目在 `twinScientist/`（小写）子目录里。Windows 文件系统大小写不敏感，最稳妥的「让 twinScientist 成为根」就是把其内容平铺到仓库根。净效果：项目文件直接位于 git 根，无嵌套。

---

## Step 1 — 处理 README 冲突（按你的选择：中文为主，英文改名）
- `git mv README.md README.en.md`（外层英文 README → README.en.md）
- `git mv twinScientist/README.md README.md`（内层中文 README → 成为根 README.md）

## Step 2 — 用 `git mv` 把受跟踪的项目文件/目录上移（保留历史）
**目录**（整体移动）：`channels config core data logs output scripts tests tools ui`
对每个执行 `git mv twinScientist/<dir> <dir>`。

**受跟踪的顶层文件**：`.env.example STARTUP.md bug_report.md main.py mc_dashboard.py requirements.txt run_real_data_research.py run_with_data.bat` — 逐个 `git mv` 上移。
（其中被修改的 `output/scientific_hypothesis_report.md` 随 `output/` 一并上移。）

**已在根目录、原地不动的外层文件**：`gen_doc.py gen_multimodal_simulator.py gen_synthetic_dalton.py create_issue.ps1 .gitignore`

## Step 3 — 用普通 `mv` 处理未跟踪文件
- `twinScientist/.env` → `.env`（含密钥，运行依赖，必须保留）
- `twinScientist/tmp_merge_check.txt` → 根（保留，非破坏性）
- 删除 `twinScientist/__pycache__` 与外层 `__pycache__`（可再生、已 gitignore）
- 删除清空后的 `twinScientist/` 空目录

## Step 4 — 修复已验证的 3 处代码断裂
1. `run_real_data_research.py:941,945` — `Path(__file__).parent.parent` → `Path(__file__).parent`（`gen_multimodal_simulator.py` 平铺后成为同级文件）
2. `tests/test_rewards.py:509,598,638` — `.parent.parent.parent` → `.parent.parent`
3. `run_with_data.bat` — 第3行 `cd /d "%~dp0.."` → `cd /d "%~dp0"`；第28行 `twinScientist\data\sensors` → `data\sensors`

> 不需改动（已确认布局无关）：`main.py`、`config/settings.py`、`config/__init__.py`、`data/organizer.py`、`mc_dashboard.py`。`core/cross_disciplinary.py` 与 `core/logic_engine.py` 有 `./data/...` 回退分支，平铺后自动走对路径，无需改。

## Step 5 — 轻量文档路径修正
- 根 `README.md`（中文）：删除 `cd twinScientist` 指令；更新「项目结构」树以反映新的扁平布局
- `STARTUP.md`：修正失效的 `TwinScientist--main3\twinScientist` 路径与 `cd twinScientist`

## Step 6 — 验证
- `git ls-files | grep ^twinScientist/`（应为空，确认无残留嵌套）
- `python -m py_compile run_real_data_research.py tests/test_rewards.py main.py`
- `python -c "import os; print(os.path.exists('gen_multimodal_simulator.py'))"`（确认同级解析）
- 若依赖就绪，可选 `python -m pytest tests/test_smoke.py -q`

## Step 7 — 提交并推送
- `git add -A`
- 提交信息：`refactor: flatten twinScientist/ into repo root`
- `git push origin main`

---

### 风险与可逆性
- 全程为「重命名 + 编辑」叠加在 main 之上，**非历史改写**，origin/main 正常 fast-forward 推送。
- 出错可通过 `git revert`/`git reset` 回滚，结构变更可逆。
- `.env` 含密钥：仅在本地文件系统移动，不进入提交（已被 .gitignore 忽略），不会外泄。

### 不会做的事
- 不删除任何源码或数据；仅删除可再生/已忽略的 `__pycache__`。
- 不运行完整应用（需 API 密钥与数据）；仅做静态编译与导入解析校验。