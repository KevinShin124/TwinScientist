@echo off
chcp 65001 >nul
cd /d "%~dp0.."

REM ============================================================
REM twinScientist 一键启动器 — 解压即用，无需额外配置
REM ============================================================

echo.
echo ============================================================
echo   twinScientist v3.0 — AI Scientist 自主科研智能体
echo ============================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.10+
    echo     下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python 已就绪
echo.

REM 检查数据目录
set SENSORS_DIR=%~dp0twinScientist\data\sensors
if not exist "%SENSORS_DIR%\*.csv" (
    echo [WARN] 未检测到传感器数据文件！
    echo.
    echo 请选择数据来源:
    echo   1. 使用内置生成器创建测试数据（推荐新手）
    echo   2. 从本地文件夹导入您的 CSV 数据
    echo   3. 跳过，仅运行空数据模式
    echo.
    set /p CHOICE="请输入选项 (1/2/3): "

    if "%CHOICE%"=="1" (
        echo [INFO] 正在生成 Daltons 格式测试数据...
        python -u gen_synthetic_dalton.py --n-samples 2000 --days 7 %*
        if errorlevel 1 (
            echo [ERROR] 数据生成失败
            pause
            exit /b 1
        )
        echo [OK] 测试数据已生成，可在 TwinScientist--main3/dalton-dataset-main/ 查看
        REM 复制到 sensors 目录
        mkdir "%SENSORS_DIR%"
        xcopy /E /Y "%~dp0dalton-dataset-main\Processed\H1\*.csv" "%SENSORS_DIR%\" 2>nul
        xcopy /E /Y "%~dp0dalton-dataset-main\Processed\H2\*.csv" "%SENSORS_DIR%\" 2>nul
        echo [OK] 数据已复制到 sensors/ 目录
    ) else if "%CHOICE%"=="2" (
        echo.
        echo 请选择包含 CSV 数据的文件夹:
        powershell -command "$folder = [System.Windows.Forms.OpenFileDialog]{new}; $folder.SelectedPath = 'C:\Users\%USERNAME%\Desktop'; $folder.ShowDialog() | Out-Null; Write-Output $folder.SelectedPath" > temp_dir.txt 2>nul
        for /f "delims=" %%i in (temp_dir.txt) do set DIR=%%i
        del temp_dir.txt >nul 2>nul

        if defined DIR (
            echo [INFO] 正在扫描 %DIR% ...
            mkdir "%SENSORS_DIR%"
            for /f "delims=" %%f in ('dir /b /s "%DIR%\*.csv"') do (
                copy /Y "%%f" "%SENSORS_DIR%\" >nul
                echo   - 已添加: %%~nxf
            )
            echo [OK] 数据导入完成
        )
    )
    echo.
)

REM 统计数据
set COUNT=0
for %%F in ("%SENSORS_DIR%\*.csv") do set /a COUNT+=1

if %COUNT%==0 (
    echo [WARN] 没有任何传感器数据文件
    echo     系统将无法进行因果推断分析
) else (
    echo [OK] 检测到 %COUNT% 个传感器数据文件
)
echo.

echo ============================================================
echo   准备启动研究引擎...
echo   按 Ctrl+C 可随时中止
echo ============================================================
echo.

REM 设置环境变量
set PYTHONPATH=%~dp0

REM 运行主程序
python -m main

pause
