@echo off
chcp 65001
echo 🧹 Очистка лишних файлов проекта...
echo.

echo 📋 Файлы для удаления:
echo.

echo ❌ Лишние .bat файлы:
if exist "install_dependencies.py" echo   - install_dependencies.py
if exist "git_commit_cleanup.bat" echo   - git_commit_cleanup.bat  
if exist "ДОБАВИТЬ_PYTHON_В_PATH.bat" echo   - ДОБАВИТЬ_PYTHON_В_PATH.bat
if exist "УСТАНОВИТЬ_БИБЛИОТЕКИ.bat" echo   - УСТАНОВИТЬ_БИБЛИОТЕКИ.bat
if exist "ИЗВЛЕЧЬ_30_КАДРОВ.bat" echo   - ИЗВЛЕЧЬ_30_КАДРОВ.bat
if exist "НАСТРОИТЬ_ОБЛАКО.bat" echo   - НАСТРОИТЬ_ОБЛАКО.bat
if exist "ОБЪЕДИНИТЬ_С_ROBOFLOW.bat" echo   - ОБЪЕДИНИТЬ_С_ROBOFLOW.bat
if exist "connect_to_github.bat" echo   - connect_to_github.bat
if exist "setup_git.bat" echo   - setup_git.bat

echo.
echo ❌ Лишние .ps1 файлы:
if exist "Commit-Cleanup.ps1" echo   - Commit-Cleanup.ps1
if exist "Add-Python-To-PATH.ps1" echo   - Add-Python-To-PATH.ps1  
if exist "setup_git.ps1" echo   - setup_git.ps1

echo.
echo ❌ Лишние .md файлы:
if exist "COMMIT_ИНСТРУКЦИЯ.md" echo   - COMMIT_ИНСТРУКЦИЯ.md
if exist "PYTHON_PATH_ШПАРГАЛКА.md" echo   - PYTHON_PATH_ШПАРГАЛКА.md
if exist "НАСТРОЙКА_PYTHON_PATH.md" echo   - НАСТРОЙКА_PYTHON_PATH.md
if exist "УСТАНОВКА_БИБЛИОТЕК.md" echo   - УСТАНОВКА_БИБЛИОТЕК.md
if exist "РУЧНОЙ_ЗАПУСК.md" echo   - РУЧНОЙ_ЗАПУСК.md
if exist "ИНСТРУКЦИЯ_30_КАДРОВ.md" echo   - ИНСТРУКЦИЯ_30_КАДРОВ.md
if exist "БЫСТРЫЙ_СТАРТ_UTAIR_CLOUD.md" echo   - БЫСТРЫЙ_СТАРТ_UTAIR_CLOUD.md
if exist "README_ROBOFLOW_PIPELINE.md" echo   - README_ROBOFLOW_PIPELINE.md
if exist "НАЧАТЬ_РАБОТУ_ROBOFLOW.md" echo   - НАЧАТЬ_РАБОТУ_ROBOFLOW.md
if exist "ФИНАЛЬНАЯ_ИНСТРУКЦИЯ_GITHUB.md" echo   - ФИНАЛЬНАЯ_ИНСТРУКЦИЯ_GITHUB.md
if exist "ШПАРГАЛКА_ВТОРОЙ_ПК.md" echo   - ШПАРГАЛКА_ВТОРОЙ_ПК.md
if exist "РАБОТА_НА_ВТОРОМ_ПК.md" echo   - РАБОТА_НА_ВТОРОМ_ПК.md
if exist "QUICK_GIT_SETUP.md" echo   - QUICK_GIT_SETUP.md
if exist "GIT_SETUP_INSTRUCTIONS.md" echo   - GIT_SETUP_INSTRUCTIONS.md

echo.
echo ❌ Лишние .py файлы:
if exist "install_libraries.py" echo   - install_libraries.py
if exist "test_python.py" echo   - test_python.py
if exist "extract_30_frames.py" echo   - extract_30_frames.py
if exist "extract_frames.py" echo   - extract_frames.py
if exist "check_cloud_sync.py" echo   - check_cloud_sync.py
if exist "ПОЛНЫЙ_PIPELINE.py" echo   - ПОЛНЫЙ_PIPELINE.py
if exist "setup_environment.py" echo   - setup_environment.py
if exist "test_environment.py" echo   - test_environment.py
if exist "simple_extractor.py" echo   - simple_extractor.py
if exist "video_to_images_ffmpeg.py" echo   - video_to_images_ffmpeg.py

echo.
set /p confirm="❓ Удалить эти файлы? (y/n): "

if /i not "%confirm%"=="y" (
    echo ⏹️ Очистка отменена
    goto end
)

echo.
echo 🗑️ Удаление файлов...

REM Удаление .bat файлов
del /q "git_commit_cleanup.bat" 2>nul
del /q "ДОБАВИТЬ_PYTHON_В_PATH.bat" 2>nul
del /q "УСТАНОВИТЬ_БИБЛИОТЕКИ.bat" 2>nul
del /q "ИЗВЛЕЧЬ_30_КАДРОВ.bat" 2>nul
del /q "НАСТРОИТЬ_ОБЛАКО.bat" 2>nul
del /q "ОБЪЕДИНИТЬ_С_ROBOFLOW.bat" 2>nul
del /q "connect_to_github.bat" 2>nul
del /q "setup_git.bat" 2>nul

REM Удаление .ps1 файлов  
del /q "Commit-Cleanup.ps1" 2>nul
del /q "Add-Python-To-PATH.ps1" 2>nul
del /q "setup_git.ps1" 2>nul

REM Удаление .md файлов
del /q "COMMIT_ИНСТРУКЦИЯ.md" 2>nul
del /q "PYTHON_PATH_ШПАРГАЛКА.md" 2>nul
del /q "НАСТРОЙКА_PYTHON_PATH.md" 2>nul
del /q "УСТАНОВКА_БИБЛИОТЕК.md" 2>nul
del /q "РУЧНОЙ_ЗАПУСК.md" 2>nul
del /q "ИНСТРУКЦИЯ_30_КАДРОВ.md" 2>nul
del /q "БЫСТРЫЙ_СТАРТ_UTAIR_CLOUD.md" 2>nul
del /q "README_ROBOFLOW_PIPELINE.md" 2>nul
del /q "НАЧАТЬ_РАБОТУ_ROBOFLOW.md" 2>nul
del /q "ФИНАЛЬНАЯ_ИНСТРУКЦИЯ_GITHUB.md" 2>nul
del /q "ШПАРГАЛКА_ВТОРОЙ_ПК.md" 2>nul
del /q "РАБОТА_НА_ВТОРОМ_ПК.md" 2>nul
del /q "QUICK_GIT_SETUP.md" 2>nul
del /q "GIT_SETUP_INSTRUCTIONS.md" 2>nul

REM Удаление .py файлов
del /q "install_libraries.py" 2>nul
del /q "test_python.py" 2>nul
del /q "extract_30_frames.py" 2>nul
del /q "extract_frames.py" 2>nul
del /q "check_cloud_sync.py" 2>nul
del /q "ПОЛНЫЙ_PIPELINE.py" 2>nul
del /q "setup_environment.py" 2>nul
del /q "test_environment.py" 2>nul
del /q "simple_extractor.py" 2>nul
del /q "video_to_images_ffmpeg.py" 2>nul
del /q "install_dependencies_python.py" 2>nul
del /q "test_python_simple.py" 2>nul

echo.
echo ✅ Очистка завершена!
echo.
echo 📊 Оставшиеся файлы проекта:
dir /b *.py *.bat *.md | findstr /v "ОЧИСТКА_ПРОЕКТА.bat"

:end
pause