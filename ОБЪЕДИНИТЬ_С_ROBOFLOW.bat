@echo off
chcp 65001 >nul
echo ========================================
echo  Объединение с Roboflow проектом
echo ========================================
echo.

REM Проверка Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git не найден!
    echo 📥 Скачайте Git с: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo ✅ Git найден

REM Настройка Git пользователя
echo 🔧 Настройка Git...
git config --global user.name "EmilioSedici16" 2>nul
git config --global user.email "emilio.sedici@example.com" 2>nul

REM Инициализация Git если нужно
if not exist .git (
    echo 🚀 Инициализация Git...
    git init
    git add .
    git commit -m "Локальные файлы: извлечение кадров для Roboflow"
) else (
    echo ℹ️  Git уже инициализирован
    echo 📝 Сохранение текущих изменений...
    git add .
    git commit -m "Обновление файлов извлечения кадров" 2>nul
)

REM Подключение к GitHub репозиторию Roboflow
echo 🔗 Подключение к GitHub репозиторию Roboflow...
git remote remove origin 2>nul
git remote add origin https://github.com/EmilioSedici16/TST_git_repository.git

echo 📥 Получение файлов Roboflow из GitHub...
git fetch origin main

echo.
echo 🔄 ОБЪЕДИНЕНИЕ ПРОЕКТОВ...
echo ========================================
echo Извлечение кадров + Roboflow = Полный pipeline
echo ========================================
echo.

REM Объединение с файлами Roboflow
git pull origin main --allow-unrelated-histories --no-edit

if %errorlevel% equ 0 (
    echo ✅ Проекты успешно объединены!
    echo.
    echo 📤 Отправка объединённого проекта в GitHub...
    git push origin main
    
    if %errorlevel% equ 0 (
        echo ✅ Объединённый проект загружен в GitHub!
        echo.
        echo 🎯 ГОТОВО! Теперь у вас есть полный pipeline:
        echo    1. Извлечение кадров из видео
        echo    2. Работа с Roboflow
        echo    3. Детекция объектов
        echo.
    ) else (
        echo ⚠️ Объединение прошло успешно, но проблема с загрузкой в GitHub
        echo 🔧 Возможно нужна аутентификация
    )
) else (
    echo ⚠️ Возникли конфликты при объединении
    echo.
    echo 🔧 Автоматическое разрешение конфликтов...
    
    REM Попытка автоматического разрешения конфликтов
    git add . 2>nul
    git commit -m "Объединение: извлечение кадров + Roboflow проект" 2>nul
    
    if %errorlevel% equ 0 (
        echo ✅ Конфликты разрешены автоматически
        git push origin main
    ) else (
        echo ❌ Требуется ручное разрешение конфликтов
        echo 📋 Выполните:
        echo    1. Откройте файлы с конфликтами
        echo    2. Разрешите конфликты вручную
        echo    3. git add .
        echo    4. git commit -m "Разрешение конфликтов"
        echo    5. git push origin main
    )
)

echo.
echo ========================================
echo        СТРУКТУРА ПРОЕКТА
echo ========================================
echo 📁 Полный pipeline компьютерного зрения:
echo.
echo 🎬 ШАГ 1: Извлечение кадров из видео
echo    ├── extract_frames.py
echo    ├── simple_extract.py  
echo    ├── Videos/ (входные видео)
echo    └── Images/ (извлеченные кадры)
echo.
echo 🤖 ШАГ 2: Работа с Roboflow
echo    ├── app.py (веб-интерфейс)
echo    ├── safety_detection.py
echo    ├── detect.py
echo    └── data/ (датасеты)
echo.
echo 🔍 ШАГ 3: Детекция и анализ
echo    ├── yolov8n.pt (модель)
echo    ├── utils.py
echo    └── requirements.txt
echo.
echo 🌐 GitHub: https://github.com/EmilioSedici16/TST_git_repository
echo.
echo 📋 Рабочий процесс:
echo    1. Поместите видео в папку Videos/
echo    2. Запустите extract_frames.py
echo    3. Загрузите кадры в Roboflow для разметки
echo    4. Обучите модель в Roboflow
echo    5. Используйте detect.py для детекции
echo.
pause