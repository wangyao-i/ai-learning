@echo off
chcp 65001 > nul
echo ========================================
echo    AI学习计划 - 微信推送快捷访问
echo ========================================
echo.
echo 请选择要访问的学习内容：
echo.
echo [1] 查看完整学习计划 (HTML版)
echo [2] 查看微信推送计划 (简洁版)
echo [3] 查看详细学习计划 (Markdown版)
echo [4] 打开Week 1学习目录
echo [5] 打开所有学习文档目录
echo.
echo [D1] 直接打开Day 1内容
echo [D2] 直接打开Day 2内容  
echo [D3] 直接打开Day 3内容
echo [D4] 直接打开Day 4内容
echo [D5] 直接打开Day 5内容
echo [D6] 直接打开Day 6内容
echo [D7] 直接打开Day 7内容
echo.
echo [W2] 打开Week 2内容
echo [W3] 打开Week 3内容
echo [W4] 打开Week 4内容
echo [W5] 打开Week 5内容
echo.
echo [Q] 退出
echo.
set /p choice=请输入选择编号: 

if "%choice%"=="1" (
    start "" "D:\1.code\ai-learning\AI学习计划-微信版.html"
    echo 正在打开HTML版学习计划...
) else if "%choice%"=="2" (
    start "" "D:\1.code\ai-learning\微信每日推送计划.md"
    echo 正在打开微信推送计划...
) else if "%choice%"=="3" (
    start "" "D:\1.code\ai-learning\AI学习计划-微信推送版.md"
    echo 正在打开详细学习计划...
) else if "%choice%"=="4" (
    start "" "D:\1.code\ai-learning\.trae\documents\Week1-LLM模型结构基础.md"
    echo 正在打开Week 1学习内容...
) else if "%choice%"=="5" (
    explorer "D:\1.code\ai-learning\.trae\documents"
    echo 正在打开学习文档目录...
) else if "%choice%"=="D1" (
    start "" "D:\1.code\ai-learning\.trae\documents\Week1-LLM模型结构基础.md#day-1-transformer整体架构"
    echo 正在打开Day 1内容...
) else if "%choice%"=="D2" (
    start "" "D:\1.code\ai-learning\.trae\documents\Week1-LLM模型结构基础.md#day-2-self-attention机制详解"
    echo 正在打开Day 2内容...
) else if "%choice%"=="D3" (
    start "" "D:\1.code\ai-learning\.trae\documents\Week1-LLM模型结构基础.md#day-3-ffnmlp结构"
    echo 正在打开Day 3内容...
) else if "%choice%"=="D4" (
    start "" "D:\1.code\ai-learning\.trae\documents\Week1-LLM模型结构基础.md#day-4-归一化层-layernormbatchnorm"
    echo 正在打开Day 4内容...
) else if "%choice%"=="D5" (
    start "" "D:\1.code\ai-learning\.trae\documents\Week1-LLM模型结构基础.md#day-5-位置编码"
    echo 正在打开Day 5内容...
) else if "%choice%"=="D6" (
    start "" "D:\1.code\ai-learning\.trae\documents\Week1-LLM模型结构基础.md#day-6-实战---手画transformer结构图"
    echo 正在打开Day 6内容...
) else if "%choice%"=="D7" (
    start "" "D:\1.code\ai-learning\.trae\documents\Week1-LLM模型结构基础.md#day-7-本周复盘--自测"
    echo 正在打开Day 7内容...
) else if "%choice%"=="W2" (
    start "" "D:\1.code\ai-learning\.trae\documents\Week2-主流LLM模型结构.md"
    echo 正在打开Week 2内容...
) else if "%choice%"=="W3" (
    start "" "D:\1.code\ai-learning\.trae\documents\Week3-LLM推理核心原理.md"
    echo 正在打开Week 3内容...
) else if "%choice%"=="W4" (
    start "" "D:\1.code\ai-learning\.trae\documents\Week4-vLLM架构深入.md"
    echo 正在打开Week 4内容...
) else if "%choice%"=="W5" (
    start "" "D:\1.code\ai-learning\.trae\documents\Week5-NPU推理深入与性能优化.md"
    echo 正在打开Week 5内容...
) else if /i "%choice%"=="Q" (
    echo 退出学习计划访问器。
) else (
    echo 无效的选择，请重新运行。
)

echo.
echo 按任意键继续...
pause > nul