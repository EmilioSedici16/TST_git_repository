@echo off
chcp 65001 >nul
echo ========================================
echo    Добавление Python в PATH
echo ========================================
echo.

echo 🐍 Поиск установленного Python...
echo.

REM Поиск Python в стандартных местах
set PYTHON_PATHS=
set FOUND_PYTHON=0

REM Проверка стандартных путей установки Python
for %%P in (
    "C:\Python*\python.exe"
    "C:\Program Files\Python*\python.exe"
    "C:\Program Files (x86)\Python*\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python*\python.exe"
    "%APPDATA%\Local\Programs\Python\Python*\python.exe"
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python*\python.exe"
    "C:\Users\%USERNAME%\AppData\Local\Microsoft\WindowsApps\python.exe"
) do (
    if exist "%%~P" (
        set PYTHON_PATH=%%~P
        set PYTHON_DIR=%%~dpP
        set FOUND_PYTHON=1
        echo ✅ Найден Python: %%~P
        goto :found
    )
)

REM Поиск через реестр Windows
echo 🔍 Поиск через реестр Windows...
for /f "tokens=2*" %%A in ('reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Python\PythonCore" /s /f "InstallPath" 2^>nul ^| findstr "InstallPath"') do (
    if exist "%%B\python.exe" (
        set PYTHON_PATH=%%B\python.exe
        set PYTHON_DIR=%%B
        set FOUND_PYTHON=1
        echo ✅ Найден Python в реестре: %%B\python.exe
        goto :found
    )
)

REM Поиск в Microsoft Store
if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" (
    set PYTHON_PATH=%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe
    set PYTHON_DIR=%LOCALAPPDATA%\Microsoft\WindowsApps
    set FOUND_PYTHON=1
    echo ✅ Найден Python из Microsoft Store
    goto :found
)

:not_found
echo ❌ Python не найден!
echo.
echo 💡 Возможные решения:
echo    1. Установите Python с https://python.org/downloads
echo    2. При установке обязательно отметьте "Add to PATH"
echo    3. Переустановите Python с правильными настройками
echo.
pause
exit /b 1

:found
echo.
echo 📍 Найденный Python:
echo    Исполняемый файл: %PYTHON_PATH%
echo    Папка: %PYTHON_DIR%
echo.

REM Проверка версии
echo 🔍 Проверка версии Python...
"%PYTHON_PATH%" --version 2>nul
if %errorlevel% neq 0 (
    echo ⚠️ Не удалось получить версию Python
) else (
    echo ✅ Python работает корректно
)
echo.

REM Проверка текущего PATH
echo 🔍 Проверка текущего PATH...
echo %PATH% | findstr /i "%PYTHON_DIR%" >nul
if %errorlevel% equ 0 (
    echo ✅ Python уже есть в PATH
    echo 💡 Возможно, нужно перезапустить терминал/компьютер
    goto :test_access
) else (
    echo ❌ Python НЕ найден в PATH
    echo 🔧 Требуется добавление в PATH
)
echo.

echo ========================================
echo      Способы добавления в PATH
echo ========================================
echo.
echo 1. Автоматическое добавление (требует прав администратора)
echo 2. Ручное добавление через системные настройки
echo 3. Временное добавление для текущей сессии
echo 4. Показать инструкцию
echo 0. Пропустить
echo.

set /p choice="Выберите способ (0-4): "

if "%choice%"=="1" goto :auto_add
if "%choice%"=="2" goto :manual_guide
if "%choice%"=="3" goto :temp_add
if "%choice%"=="4" goto :show_guide
if "%choice%"=="0" goto :end
goto :invalid_choice

:auto_add
echo.
echo 🔧 АВТОМАТИЧЕСКОЕ ДОБАВЛЕНИЕ В PATH
echo ========================================
echo.

echo ⚠️ Требуются права администратора!
echo 📝 Будет добавлено в системный PATH:
echo    %PYTHON_DIR%
echo    %PYTHON_DIR%Scripts
echo.

set /p confirm="Продолжить? (y/n): "
if /i not "%confirm%"=="y" goto :manual_guide

echo 🚀 Добавление в PATH...

REM Получить текущий PATH
for /f "tokens=2*" %%A in ('reg query "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul') do set CURRENT_PATH=%%B

REM Добавить Python пути
set NEW_PATH=%CURRENT_PATH%;%PYTHON_DIR%;%PYTHON_DIR%Scripts

REM Записать в реестр
reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH /t REG_EXPAND_SZ /d "%NEW_PATH%" /f >nul 2>&1

if %errorlevel% equ 0 (
    echo ✅ Python добавлен в системный PATH
    echo 🔄 Перезапустите компьютер для применения изменений
    
    REM Уведомить систему об изменении
    setx PATH "%NEW_PATH%" /M >nul 2>&1
    
    echo ✅ Изменения применены
) else (
    echo ❌ Ошибка добавления в PATH
    echo 💡 Возможно, недостаточно прав администратора
    echo 🔧 Попробуйте ручной способ
    goto :manual_guide
)

goto :test_access

:temp_add
echo.
echo ⚡ ВРЕМЕННОЕ ДОБАВЛЕНИЕ В PATH
echo ========================================
echo.

echo 🔧 Добавление для текущей сессии...
set PATH=%PATH%;%PYTHON_DIR%;%PYTHON_DIR%Scripts

echo ✅ Python временно добавлен в PATH
echo ⚠️ Действует только до закрытия окна
echo.

goto :test_access

:manual_guide
echo.
echo 🔧 РУЧНОЕ ДОБАВЛЕНИЕ В PATH
echo ========================================
echo.
echo 📋 Пошаговая инструкция:
echo.
echo 1. Нажмите Win + R
echo 2. Введите: sysdm.cpl
echo 3. Нажмите "Переменные среды"
echo 4. В "Системные переменные" найдите PATH
echo 5. Нажмите "Изменить"
echo 6. Нажмите "Создать" и добавьте:
echo    %PYTHON_DIR%
echo 7. Нажмите "Создать" и добавьте:
echo    %PYTHON_DIR%Scripts
echo 8. Нажмите OK во всех окнах
echo 9. Перезапустите компьютер
echo.

echo 📋 Пути для добавления:
echo ┌─────────────────────────────────────────────┐
echo │ %PYTHON_DIR%
echo │ %PYTHON_DIR%Scripts
echo └─────────────────────────────────────────────┘
echo.

REM Создать файл с путями для копирования
echo %PYTHON_DIR%> python_paths.txt
echo %PYTHON_DIR%Scripts>> python_paths.txt
echo ✅ Пути сохранены в файл python_paths.txt для копирования
echo.

set /p manual_done="Выполнили ручное добавление? (y/n): "
if /i "%manual_done%"=="y" goto :test_access

goto :show_guide

:show_guide
echo.
echo 📚 ПОДРОБНАЯ ИНСТРУКЦИЯ
echo ========================================
echo.
echo 🎯 Цель: Сделать Python доступным из любой папки
echo.
echo 🔧 Метод 1 - Через Windows UI:
echo    1. Win + X → "Система"
echo    2. "Дополнительные параметры системы"  
echo    3. "Переменные среды"
echo    4. Системные переменные → PATH → Изменить
echo    5. Добавить новые записи:
echo       • %PYTHON_DIR%
echo       • %PYTHON_DIR%Scripts
echo.
echo 🔧 Метод 2 - Через командную строку (от администратора):
echo    setx PATH "%%PATH%%;%PYTHON_DIR%;%PYTHON_DIR%Scripts" /M
echo.
echo 🔧 Метод 3 - Переустановка Python:
echo    1. Скачать Python с python.org
echo    2. При установке отметить "Add to PATH"
echo    3. Выбрать "Customize installation"
echo    4. Отметить "Add Python to environment variables"
echo.

goto :end

:test_access
echo.
echo 🧪 ТЕСТИРОВАНИЕ ДОСТУПА К PYTHON
echo ========================================
echo.

echo 🔍 Проверка доступности python...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ python команда работает
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo    Версия: %%i
) else (
    echo ❌ python команда недоступна
    echo 💡 Перезапустите терминал или компьютер
)

echo.
echo 🔍 Проверка доступности pip...
python -m pip --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ pip команда работает
    for /f "tokens=*" %%i in ('python -m pip --version 2^>^&1') do echo    %%i
) else (
    echo ❌ pip команда недоступна
)

echo.
echo 🔍 Проверка PATH...
echo %PATH% | findstr /i "%PYTHON_DIR%" >nul
if %errorlevel% equ 0 (
    echo ✅ Python пути найдены в PATH
) else (
    echo ❌ Python пути НЕ найдены в PATH
    echo 🔄 Требуется перезапуск терминала/компьютера
)

echo.
echo 🧪 Тест установки библиотек...
python -c "print('✅ Python импорты работают')" 2>nul
if %errorlevel% equ 0 (
    echo ✅ Python скрипты выполняются
    echo 🚀 Можно устанавливать библиотеки!
    echo.
    echo 💡 Следующие шаги:
    echo    1. УСТАНОВИТЬ_БИБЛИОТЕКИ.bat
    echo    2. test_python.py  
    echo    3. ИЗВЛЕЧЬ_30_КАДРОВ.bat
) else (
    echo ❌ Проблемы с выполнением Python скриптов
)

goto :end

:invalid_choice
echo ❌ Неверный выбор. Попробуйте снова.
goto :choice

:end
echo.
echo ========================================
echo              ИТОГИ
echo ========================================
echo.

if %FOUND_PYTHON% equ 1 (
    echo ✅ Python найден: %PYTHON_PATH%
    echo 📁 Папка: %PYTHON_DIR%
    echo.
    echo 🔄 Если команды python/pip не работают:
    echo    1. Перезапустите терминал
    echo    2. Перезапустите Cursor
    echo    3. Перезагрузите компьютер
    echo.
    echo 📋 Файлы для справки:
    echo    • python_paths.txt - пути для ручного добавления
    echo    • УСТАНОВКА_БИБЛИОТЕК.md - инструкция по библиотекам
    echo.
    echo 🚀 После настройки PATH запустите:
    echo    УСТАНОВИТЬ_БИБЛИОТЕКИ.bat
) else (
    echo ❌ Python не найден в системе
    echo 💡 Установите Python с https://python.org/downloads
)

echo.
pause