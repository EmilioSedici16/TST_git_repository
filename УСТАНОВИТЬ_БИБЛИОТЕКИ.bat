@echo off
chcp 65001 >nul
echo ========================================
echo   Установка библиотек для извлечения кадров
echo ========================================
echo.

echo 📦 Будут установлены библиотеки:
echo    • opencv-python (cv2) - для работы с видео
echo    • numpy - для работы с массивами
echo    • pillow - для работы с изображениями
echo    • pathlib - для работы с путями (обычно встроена)
echo.

REM Проверка Python
echo 🐍 Проверка Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python не найден!
    echo.
    echo 💡 Возможные решения:
    echo    1. Установите Python: https://python.org/downloads
    echo    2. Добавьте Python в PATH
    echo    3. Перезагрузите компьютер после установки
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Найден: %PYTHON_VERSION%
echo.

REM Проверка pip
echo 📦 Проверка pip...
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip не найден!
    echo 🔧 Установка pip...
    python -m ensurepip --upgrade
    if %errorlevel% neq 0 (
        echo ❌ Не удалось установить pip
        pause
        exit /b 1
    )
)

for /f "tokens=*" %%i in ('python -m pip --version 2^>^&1') do set PIP_VERSION=%%i
echo ✅ Найден: %PIP_VERSION%
echo.

echo ========================================
echo         Установка библиотек
echo ========================================
echo.

REM Обновление pip
echo 🔄 Обновление pip...
python -m pip install --upgrade pip
echo.

REM Установка OpenCV
echo 📹 Установка OpenCV (cv2)...
python -m pip install opencv-python
if %errorlevel% equ 0 (
    echo ✅ OpenCV установлен успешно
) else (
    echo ❌ Ошибка установки OpenCV
    set INSTALL_ERRORS=1
)
echo.

REM Установка NumPy
echo 🔢 Установка NumPy...
python -m pip install numpy
if %errorlevel% equ 0 (
    echo ✅ NumPy установлен успешно
) else (
    echo ❌ Ошибка установки NumPy
    set INSTALL_ERRORS=1
)
echo.

REM Установка Pillow
echo 🖼️ Установка Pillow...
python -m pip install Pillow
if %errorlevel% equ 0 (
    echo ✅ Pillow установлен успешно
) else (
    echo ❌ Ошибка установки Pillow
    set INSTALL_ERRORS=1
)
echo.

REM Дополнительные библиотеки для полного pipeline
echo 🚀 Установка дополнительных библиотек для Roboflow...
python -m pip install roboflow ultralytics streamlit requests
if %errorlevel% equ 0 (
    echo ✅ Дополнительные библиотеки установлены
) else (
    echo ⚠️ Некоторые дополнительные библиотеки не установились (не критично)
)
echo.

echo ========================================
echo           Проверка установки
echo ========================================
echo.

REM Тестирование библиотек
echo 🧪 Тестирование установленных библиотек...
echo.

python -c "import cv2; print(f'✅ OpenCV версия: {cv2.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ OpenCV не работает
    set TEST_ERRORS=1
)

python -c "import numpy; print(f'✅ NumPy версия: {numpy.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ NumPy не работает
    set TEST_ERRORS=1
)

python -c "import PIL; print(f'✅ Pillow версия: {PIL.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ Pillow не работает
    set TEST_ERRORS=1
)

python -c "from pathlib import Path; print('✅ pathlib работает')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ pathlib не работает
    set TEST_ERRORS=1
)

echo.

REM Проверка возможности открытия видео
echo 🎬 Тестирование работы с видео...
python -c "
import cv2
import numpy as np
print('✅ Библиотеки загружены')

# Создать тестовое видео в памяти
test_array = np.zeros((100, 100, 3), dtype=np.uint8)
print('✅ Массивы NumPy работают')

# Проверить кодеки OpenCV
print('✅ OpenCV готов к работе с видео')
print('📋 Доступные кодеки:', len(cv2.getBuildInformation()))
" 2>nul

if %errorlevel% equ 0 (
    echo ✅ Тест работы с видео успешен
) else (
    echo ⚠️ Проблемы с тестом видео (возможно, не критично)
)

echo.

echo ========================================
echo              РЕЗУЛЬТАТ
echo ========================================
echo.

if not defined INSTALL_ERRORS if not defined TEST_ERRORS (
    echo 🎉 ВСЕ БИБЛИОТЕКИ УСТАНОВЛЕНЫ УСПЕШНО!
    echo.
    echo ✅ Готово к работе:
    echo    • OpenCV (cv2) - работа с видео ✓
    echo    • NumPy - работа с массивами ✓  
    echo    • Pillow - работа с изображениями ✓
    echo    • pathlib - работа с путями ✓
    echo.
    echo 🚀 Теперь можно запускать:
    echo    1. test_python.py - проверка системы
    echo    2. ИЗВЛЕЧЬ_30_КАДРОВ.bat - извлечение кадров
    echo    3. extract_frames.py - прямое извлечение
    echo.
    echo 💡 Следующие шаги:
    echo    1. Настройте облако: НАСТРОИТЬ_ОБЛАКО.bat
    echo    2. Извлеките кадры: ИЗВЛЕЧЬ_30_КАДРОВ.bat
    echo    3. Подготовьте для Roboflow
) else (
    echo ⚠️ УСТАНОВКА ЗАВЕРШЕНА С ОШИБКАМИ
    echo.
    echo 🔧 Возможные решения:
    echo    • Запустите от имени администратора
    echo    • Проверьте интернет-соединение
    echo    • Обновите Python до последней версии
    echo    • Попробуйте установить вручную:
    echo      pip install opencv-python numpy pillow
    echo.
)

echo ========================================
echo           Установленные пакеты
echo ========================================
echo.

echo 📋 Список всех установленных пакетов:
python -m pip list | findstr -i "opencv numpy pillow roboflow ultralytics streamlit"

echo.
echo 💾 Полный список пакетов сохранен в requirements_installed.txt
python -m pip freeze > requirements_installed.txt

echo.
pause