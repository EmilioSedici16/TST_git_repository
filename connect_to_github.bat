@echo off
chcp 65001 >nul
echo ========================================
echo   Подключение к GitHub репозиторию
echo ========================================

REM Проверка Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git не найден!
    echo 📥 Скачайте Git с: https://git-scm.com/download/win
    echo 🔄 После установки перезапустите этот файл
    pause
    exit /b 1
)

echo ✅ Git найден

REM Настройка Git пользователя
echo 🔧 Настройка Git пользователя...
git config --global user.name "EmilioSedici16" 2>nul
git config --global user.email "emilio.sedici@example.com" 2>nul

REM Инициализация если нужно
if not exist .git (
    echo 🚀 Инициализация Git...
    git init
    git add .
    git commit -m "Первый коммит: локальные файлы TST_Project"
) else (
    echo ℹ️  Git уже инициализирован
    echo 📝 Добавление текущих изменений...
    git add .
    git commit -m "Обновление локальных файлов перед подключением к GitHub" 2>nul
)

REM Подключение к GitHub
echo 🔗 Подключение к GitHub репозиторию...
git remote remove origin 2>nul
git remote add origin https://github.com/EmilioSedici16/TST_git_repository.git

REM Проверка подключения
echo 🔍 Проверка подключения...
git remote -v

REM Получение информации о удалённом репозитории
echo 📥 Получение информации из GitHub...
git fetch origin main 2>nul

echo.
echo ========================================
echo        Выберите действие:
echo ========================================
echo 1. Отправить МОИ локальные файлы в GitHub
echo    (заменит содержимое GitHub на ваши файлы)
echo.
echo 2. Получить файлы ИЗ GitHub и объединить
echo    (объединит файлы из GitHub с вашими)
echo.
echo 3. Только подключиться (не синхронизировать)
echo.
set /p choice="Введите номер (1, 2 или 3): "

if "%choice%"=="1" (
    echo.
    echo 📤 Отправка локальных файлов в GitHub...
    echo ⚠️  Это заменит содержимое GitHub репозитория!
    set /p confirm="Продолжить? (y/n): "
    if /i "%confirm%"=="y" (
        git branch -M main
        git push -u origin main --force
        if %errorlevel% equ 0 (
            echo ✅ Локальные файлы успешно отправлены в GitHub!
        ) else (
            echo ❌ Ошибка при отправке. Возможные причины:
            echo    - Нет прав доступа к репозиторию
            echo    - Нужна аутентификация через Personal Access Token
            echo    - Проблемы с интернет-соединением
        )
    ) else (
        echo ❌ Операция отменена пользователем
    )
) else if "%choice%"=="2" (
    echo.
    echo 📥 Получение и объединение файлов из GitHub...
    git pull origin main --allow-unrelated-histories
    if %errorlevel% equ 0 (
        echo ✅ Файлы успешно объединены!
        echo 📤 Отправка объединённых файлов обратно в GitHub...
        git push origin main
        if %errorlevel% equ 0 (
            echo ✅ Синхронизация завершена успешно!
        ) else (
            echo ⚠️ Объединение прошло успешно, но возникла проблема с отправкой.
        )
    ) else (
        echo ⚠️ Возникли конфликты при объединении.
        echo 🔧 Разрешите их вручную и выполните:
        echo    git add .
        echo    git commit -m "Разрешение конфликтов"
        echo    git push origin main
    )
) else if "%choice%"=="3" (
    echo ✅ Репозиторий подключён без синхронизации
    echo 📋 Для синхронизации выполните вручную:
    echo    git pull origin main --allow-unrelated-histories
    echo    git push origin main
) else (
    echo ❌ Неверный выбор. Перезапустите скрипт.
    pause
    exit /b 1
)

echo.
echo ========================================
echo          Настройка завершена!
echo ========================================
echo 🌐 Ваш репозиторий: https://github.com/EmilioSedici16/TST_git_repository
echo.
echo 📋 Для ежедневной работы используйте:
echo.
echo    ПЕРЕД работой:
echo    git pull origin main
echo.
echo    ПОСЛЕ работы:
echo    git add .
echo    git commit -m "Описание изменений"
echo    git push origin main
echo.
echo 📖 Подробные инструкции: ПОДКЛЮЧЕНИЕ_К_СУЩЕСТВУЮЩЕМУ_REPO.md
echo 💻 Работа на втором ПК: РАБОТА_НА_ВТОРОМ_ПК.md
echo.
pause