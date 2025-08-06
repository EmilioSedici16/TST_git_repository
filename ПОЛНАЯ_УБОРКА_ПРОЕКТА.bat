@echo off
chcp 65001
echo 🧹 Полная уборка проекта...
echo.

echo 📋 1. Удаление временных файлов Python:
if exist "__pycache__" (
    echo Удаляем __pycache__...
    rmdir /s /q "__pycache__"
    echo ✅ __pycache__ удален
) else (
    echo ℹ️ __pycache__ не найден
)

echo.
echo 📋 2. Удаление файлов очистки (они уже не нужны):
del /q "ОЧИСТКА_ПРОЕКТА.bat" 2>nul && echo ✅ ОЧИСТКА_ПРОЕКТА.bat удален
del /q "ФИНАЛЬНАЯ_ОЧИСТКА.bat" 2>nul && echo ✅ ФИНАЛЬНАЯ_ОЧИСТКА.bat удален
del /q "ДИАГНОСТИКА_ПРЕФИКСА.bat" 2>nul && echo ✅ ДИАГНОСТИКА_ПРЕФИКСА.bat удален

echo.
echo 📋 3. Удаление дублирующихся файлов:
del /q "install_dependencies.py" 2>nul && echo ✅ install_dependencies.py удален
del /q "cursorrules" 2>nul && echo ✅ cursorrules удален

echo.
echo 📋 4. Проверка структуры папок:
echo 📁 Основные папки:
for /d %%d in (*) do (
    if /i not "%%d"=="__pycache__" (
        if /i not "%%d"==".git" (
            if /i not "%%d"==".cursor" (
                if /i not "%%d"==".vscode" (
                    echo   ✅ %%d
                )
            )
        )
    )
)

echo.
echo 📋 5. Итоговые файлы проекта:
echo.
echo 🐍 Python файлы:
dir /b *.py 2>nul | find /v /c "" >nul && (
    for %%f in (*.py) do echo   ✅ %%f
) || echo   ℹ️ Python файлы не найдены

echo.
echo 🛠️ Скрипты и утилиты:
dir /b *.bat 2>nul | find /v /c "" >nul && (
    for %%f in (*.bat) do (
        if /i not "%%f"=="ПОЛНАЯ_УБОРКА_ПРОЕКТА.bat" echo   ✅ %%f
    )
) || echo   ℹ️ Bat файлы не найдены

echo.
echo 📚 Документация:
dir /b *.md 2>nul | find /v /c "" >nul && (
    for %%f in (*.md) do echo   ✅ %%f
) || echo   ℹ️ Markdown файлы не найдены

echo.
echo 📊 Статистика после уборки:
echo ================================
for /f %%i in ('dir /b *.py 2^>nul ^| find /c /v ""') do echo Python файлов: %%i
for /f %%i in ('dir /b *.bat 2^>nul ^| find /c /v ""') do echo Bat файлов: %%i  
for /f %%i in ('dir /b *.md 2^>nul ^| find /c /v ""') do echo Markdown файлов: %%i

echo.
echo ✅ Полная уборка проекта завершена!
echo 🎯 Проект готов к продуктивной работе!

pause