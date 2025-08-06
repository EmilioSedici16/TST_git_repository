@echo off
chcp 65001
echo 🧹 Финальная очистка проекта...
echo.

echo 📋 Дополнительные файлы для удаления:
echo.

echo ❌ Дублирующиеся/устаревшие файлы:
if exist "install_dependencies.py" echo   - install_dependencies.py ^(заменен на УСТАНОВКА_ЗАВИСИМОСТЕЙ.bat^)
if exist "cursorrules" echo   - cursorrules ^(дублирует .cursorrules^)
if exist "КОММИТ_ПРОГРЕССА.bat" echo   - КОММИТ_ПРОГРЕССА.bat ^(использован, больше не нужен^)

echo.
echo ❌ Временные bat-файлы:
if exist "ИСПРАВИТЬ_ТЕРМИНАЛ.bat" echo   - ИСПРАВИТЬ_ТЕРМИНАЛ.bat ^(не сработал^)
if exist "БЫСТРЫЙ_ТЕСТ.bat" echo   - БЫСТРЫЙ_ТЕСТ.bat ^(функции есть в БЫСТРЫЕ_КОМАНДЫ.bat^)
if exist "ПРОВЕРКА_ВИДЕО.bat" echo   - ПРОВЕРКА_ВИДЕО.bat ^(функции есть в БЫСТРЫЕ_КОМАНДЫ.bat^)
if exist "НАСТРОЙКА_UTAIR_CLOUD.bat" echo   - НАСТРОЙКА_UTAIR_CLOUD.bat ^(видео уже скачаны локально^)

echo.
echo ❌ Статусные файлы (после завершения настройки):
if exist "СТАТУС_ПРОЕКТА_ПК2.md" echo   - СТАТУС_ПРОЕКТА_ПК2.md ^(настройка завершена^)
if exist "ИНСТРУКЦИЯ_ПОСЛЕ_ПЕРЕЗАГРУЗКИ.md" echo   - ИНСТРУКЦИЯ_ПОСЛЕ_ПЕРЕЗАГРУЗКИ.md ^(настройка завершена^)

echo.
echo ✅ Останется чистый проект:
echo   🐍 Основные Python файлы ^(app.py, detect.py, safety_detection.py, utils.py, video_to_images.py, test_basic.py^)
echo   📄 Конфигурация ^(requirements.txt, .cursorrules, .gitignore^)
echo   📚 Документация ^(README.md, ROBOFLOW_SETUP.md, QUICKSTART.md, СИНХРОНИЗАЦИЯ_UTAIR_CLOUD.md^)
echo   🛠️ Полезные утилиты ^(БЫСТРЫЕ_КОМАНДЫ.bat, diagnostic_script.py, cmd_helper.py^)
echo   🧪 Диагностика ^(ПРОВЕРКА_УСТАНОВКИ.bat, ТЕСТ_ПРОЕКТА.bat, УСТАНОВКА_ЗАВИСИМОСТЕЙ.bat^)
echo   🎬 Модель и данные ^(yolov8n.pt, Videos/, Images/, data/^)

echo.
set /p confirm="❓ Провести финальную очистку? (y/n): "

if /i not "%confirm%"=="y" (
    echo ⏹️ Очистка отменена
    goto end
)

echo.
echo 🗑️ Удаление файлов...

del /q "install_dependencies.py" 2>nul
del /q "cursorrules" 2>nul
del /q "КОММИТ_ПРОГРЕССА.bat" 2>nul
del /q "ИСПРАВИТЬ_ТЕРМИНАЛ.bat" 2>nul
del /q "БЫСТРЫЙ_ТЕСТ.bat" 2>nul
del /q "ПРОВЕРКА_ВИДЕО.bat" 2>nul
del /q "НАСТРОЙКА_UTAIR_CLOUD.bat" 2>nul
del /q "СТАТУС_ПРОЕКТА_ПК2.md" 2>nul
del /q "ИНСТРУКЦИЯ_ПОСЛЕ_ПЕРЕЗАГРУЗКИ.md" 2>nul

echo.
echo ✅ Финальная очистка завершена!
echo.
echo 📊 Итоговый состав проекта:
echo.
echo 🐍 Python файлы:
dir /b *.py 2>nul
echo.
echo 🛠️ Скрипты и утилиты:
dir /b *.bat 2>nul
echo.
echo 📚 Документация:
dir /b *.md 2>nul
echo.
echo 📁 Папки:
dir /ad /b 2>nul | findstr /v "__pycache__"

:end
pause