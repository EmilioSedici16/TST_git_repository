@echo off
chcp 65001 >nul
echo ========================================
echo    Настройка Git для TST_Project
echo ========================================
echo.

REM Проверка наличия Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git не найден!
    echo 📥 Скачайте Git с: https://git-scm.com/download/win
    echo 🔄 После установки перезапустите этот файл
    pause
    exit /b 1
)

echo ✅ Git найден

REM Проверка, инициализирован ли уже Git
if exist .git (
    echo ℹ️  Git репозиторий уже инициализирован
    goto :push_check
)

echo 🚀 Инициализация Git репозитория...
git init

echo 📁 Добавление файлов в репозиторий...
git add .

echo 💾 Создание первого коммита...
git commit -m "Первый коммит: настройка проекта TST_Project"

:push_check
echo.
echo ========================================
echo       Следующие шаги:
echo ========================================
echo 1. Создайте репозиторий на GitHub:
echo    - Перейдите на https://github.com
echo    - Нажмите "New repository"
echo    - Назовите: TST_Project
echo    - НЕ добавляйте README, .gitignore
echo    - Нажмите "Create repository"
echo.
echo 2. Скопируйте ваш GitHub username
set /p username="   Введите ваш GitHub username: "

echo.
echo 🔗 Подключение к GitHub...
git remote add origin https://github.com/%username%/TST_Project.git
git branch -M main

echo.
echo 📤 Отправка кода на GitHub...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ✅ Успешно! Репозиторий настроен
    echo 🌐 Ваш проект: https://github.com/%username%/TST_Project
    echo.
    echo 📋 Для ежедневной работы используйте:
    echo    git pull origin main    ^(перед работой^)
    echo    git add .
    echo    git commit -m "описание"
    echo    git push origin main    ^(после работы^)
) else (
    echo.
    echo ❌ Ошибка при отправке на GitHub
    echo 🔧 Проверьте:
    echo    - Правильность username
    echo    - Создан ли репозиторий на GitHub
    echo    - Настроена ли аутентификация Git
)

echo.
pause