@echo off
chcp 65001
echo 📁 Организация структуры проекта компьютерного зрения...
echo.

echo 🎯 Создаем оптимальную структуру папок:
echo.

REM Создание основных папок
echo 📂 Создание папок для организации:

if not exist "scripts" mkdir "scripts" && echo ✅ scripts/ - вспомогательные скрипты
if not exist "docs" mkdir "docs" && echo ✅ docs/ - документация
if not exist "models" mkdir "models" && echo ✅ models/ - модели машинного обучения
if not exist "results" mkdir "results" && echo ✅ results/ - результаты анализа
if not exist "config" mkdir "config" && echo ✅ config/ - конфигурационные файлы

echo.
echo 🔄 Перемещение файлов в соответствующие папки:

REM Перемещение скриптов
echo 📝 Скрипты и утилиты:
move "БЫСТРЫЕ_КОМАНДЫ.bat" "scripts\" 2>nul && echo   ✅ БЫСТРЫЕ_КОМАНДЫ.bat → scripts/
move "ПРОВЕРКА_УСТАНОВКИ.bat" "scripts\" 2>nul && echo   ✅ ПРОВЕРКА_УСТАНОВКИ.bat → scripts/
move "ТЕСТ_ПРОЕКТА.bat" "scripts\" 2>nul && echo   ✅ ТЕСТ_ПРОЕКТА.bat → scripts/
move "УСТАНОВКА_ЗАВИСИМОСТЕЙ.bat" "scripts\" 2>nul && echo   ✅ УСТАНОВКА_ЗАВИСИМОСТЕЙ.bat → scripts/
move "ОТКРЫТЬ_VS_CODE.bat" "scripts\" 2>nul && echo   ✅ ОТКРЫТЬ_VS_CODE.bat → scripts/
move "cmd_helper.py" "scripts\" 2>nul && echo   ✅ cmd_helper.py → scripts/
move "diagnostic_script.py" "scripts\" 2>nul && echo   ✅ diagnostic_script.py → scripts/

REM Перемещение документации
echo 📚 Документация:
move "ROBOFLOW_SETUP.md" "docs\" 2>nul && echo   ✅ ROBOFLOW_SETUP.md → docs/
move "QUICKSTART.md" "docs\" 2>nul && echo   ✅ QUICKSTART.md → docs/
move "СИНХРОНИЗАЦИЯ_UTAIR_CLOUD.md" "docs\" 2>nul && echo   ✅ СИНХРОНИЗАЦИЯ_UTAIR_CLOUD.md → docs/

REM Перемещение модели
echo 🤖 Модели:
move "yolov8n.pt" "models\" 2>nul && echo   ✅ yolov8n.pt → models/

REM Перемещение конфигурации  
echo ⚙️ Конфигурация:
move "requirements.txt" "config\" 2>nul && echo   ✅ requirements.txt → config/
copy ".cursorrules" "config\cursorrules_backup.txt" 2>nul && echo   ✅ .cursorrules → config/ (копия)

echo.
echo 📁 Создание ярлыков для быстрого доступа:

REM Создание ярлыков в корне
echo @echo off > "БЫСТРОЕ_МЕНЮ.bat"
echo cd /d "%%~dp0" >> "БЫСТРОЕ_МЕНЮ.bat"
echo call "scripts\БЫСТРЫЕ_КОМАНДЫ.bat" >> "БЫСТРОЕ_МЕНЮ.bat"

echo @echo off > "ЗАПУСК_STREAMLIT.bat"
echo cd /d "%%~dp0" >> "ЗАПУСК_STREAMLIT.bat"
echo echo 🌐 Запуск веб-интерфейса... >> "ЗАПУСК_STREAMLIT.bat"
echo streamlit run app.py >> "ЗАПУСК_STREAMLIT.bat"

echo ✅ БЫСТРОЕ_МЕНЮ.bat - ярлык для главного меню
echo ✅ ЗАПУСК_STREAMLIT.bat - ярлык для веб-интерфейса

echo.
echo 📊 Итоговая структура проекта:
echo ================================
echo.
echo 📁 TST_git_repository/
echo ├── 🐍 Основные Python модули:
for %%f in (app.py detect.py safety_detection.py utils.py video_to_images.py test_basic.py) do (
    if exist "%%f" echo │   ├── %%f
)
echo │
echo ├── 📂 scripts/           # Скрипты и утилиты
echo ├── 📂 docs/              # Документация  
echo ├── 📂 models/            # Модели ML
echo ├── 📂 config/            # Конфигурация
echo ├── 📂 Videos/            # Исходные видео
echo ├── 📂 Images/            # Извлеченные кадры
echo ├── 📂 data/              # Тестовые данные
echo ├── 📂 results/           # Результаты анализа
echo │
echo ├── ⚡ БЫСТРОЕ_МЕНЮ.bat    # Главное меню
echo ├── 🌐 ЗАПУСК_STREAMLIT.bat # Веб-интерфейс
echo └── 📄 README.md          # Основная документация

echo.
echo ✅ Организация проекта завершена!
echo 🎯 Теперь проект имеет четкую структуру для эффективной работы
echo.
echo 🚀 Для работы используйте:
echo    - БЫСТРОЕ_МЕНЮ.bat - главное меню всех функций
echo    - ЗАПУСК_STREAMLIT.bat - веб-интерфейс
echo    - scripts/ - все вспомогательные скрипты
echo    - docs/ - вся документация

pause