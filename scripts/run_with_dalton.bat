@echo off
REM ============================================================
REM twinScientist + DALTON Dataset — 一键启动脚本
REM ============================================================
REM
REM 使用说明:
REM   1. 将下载的 DALTON CSV 文件放到 data/sensors/ 目录
REM      推荐: Merged/data_H1.csv ~ data_H8.csv
REM   2. 双击运行此脚本，或从命令行执行
REM   3. Agent 会自动检测数据格式并加载数据进行因果推断
REM ============================================================

cd /d "%~dp0"

echo ========================================
echo   twinScientist + DALTON 研究管道
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo [1/3] 检查依赖...
pip install -q -r requirements.txt 2>nul || echo [提示] 某些依赖可能未安装，将继续尝试...

echo [2/3] 验证数据目录...
if not exist "data\sensors" mkdir "data\sensors"

REM Auto-generate synthetic Daltons test data if no CSV found
set csv_count=0
for %%F in (data\sensors\*.csv) do set /a csv_count+=1

if %csv_count% equ 0 (
    echo [3/3] 未检测到传感器数据，生成合成 Daltons 测试数据...
    python -c "^
import sys; sys.path.insert(0, '.');^
from data.dalton_ingest import generate_synthetic_dalton_data;^
import os;^
os.makedirs('data/sensors', exist_ok=True);^
synth = generate_synthetic_dalton_data(n_points_per_device=500, houses=['H1','H2'], rooms=['Kitchen','Bedroom']);^
file_count = 0;^
for house, room_groups in synth.items():^
    for room_files in room_groups:^
        for records, fname in room_files:^
            out_path = f'data/sensors/{house}_{fname}';^
            if records:^
                cols = list(records[0].keys());^
                with open(out_path, 'w', encoding='utf-8') as f:^
                    f.write(','.join(cols)+'\n');^
                    for rec in records:^
                        vals = ','.join(str(rec[c]) for c in cols);^
                        f.write(vals+'\n');^
                file_count += 1;^
print(f'[DaltonIngest] Generated {file_count} test files');^
"
) else (
    echo [3/3] 检测到 %csv_count% 个 CSV 数据文件，直接启动 agent...
)

echo.
echo [启动] 正在启动 twinScientist CLI...
echo ----------------------------------------
echo 研究问题: Dalton dataset indoor air quality causal study
echo 领域:     Indoor Air Quality & Human Health
echo 迭代次数: 5（含 1 次反思循环）
echo ----------------------------------------
echo.

REM Launch the agent with Daltons research question
python main.py ^
    --question "Dalton dataset indoor air quality causal study: How do temperature and CO2 levels causally affect indoor pollutant concentrations in low-to-middle-income Indian households?" ^
    --domain "Indoor Air Quality & Human Health" ^
    --iterations 5

echo.
echo [完成] 研究已完成！查看 output/ 目录获取报告。
pause
