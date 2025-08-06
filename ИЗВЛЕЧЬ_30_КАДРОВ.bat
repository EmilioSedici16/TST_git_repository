@echo off
chcp 65001 >nul
echo ========================================
echo   Извлечение 30 кадров из каждого видео
echo ========================================
echo.

echo 🎯 Параметры извлечения:
echo    • 30 кадров из каждого видео
echo    • Равномерное распределение по времени
echo    • Первый кадр из каждого сегмента
echo    • Сохранение в облако UTair
echo.

REM Проверка Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python не найден!
    echo 📥 Установите Python: https://python.org/downloads
    pause
    exit /b 1
)

echo ✅ Python найден

REM Проверка папки Videos
if exist Videos\ (
    echo ✅ Папка Videos найдена
    
    REM Подсчет видеофайлов
    set /a count=0
    for %%f in (Videos\*.mp4 Videos\*.avi Videos\*.mov Videos\*.mkv) do set /a count+=1
    
    if %count% GTR 0 (
        echo 📹 Найдено видеофайлов: %count%
        echo 🖼️ Будет извлечено кадров: %count% × 30 = %count%0
    ) else (
        echo ⚠️ Видеофайлы не найдены в папке Videos
        echo 🌐 Проверьте облако: https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos
        echo 🔧 Настройте доступ: НАСТРОИТЬ_ОБЛАКО.bat
        pause
        exit /b 1
    )
) else (
    echo ❌ Папка Videos не найдена
    echo 🔧 Настройте облако: НАСТРОИТЬ_ОБЛАКО.bat
    pause
    exit /b 1
)

echo.
echo ========================================
echo        Выберите режим работы
echo ========================================
echo.
echo 1. Быстрое извлечение (обновленный extract_frames.py)
echo 2. Продвинутое извлечение (новый extract_30_frames.py)
echo 3. Настроить облако перед извлечением
echo 4. Проверить синхронизацию с облаком
echo 0. Отмена
echo.

set /p choice="Выберите режим (0-4): "

if "%choice%"=="1" goto :quick_extract
if "%choice%"=="2" goto :advanced_extract
if "%choice%"=="3" goto :setup_cloud
if "%choice%"=="4" goto :check_sync
if "%choice%"=="0" goto :cancel
goto :invalid_choice

:quick_extract
echo.
echo 🚀 БЫСТРОЕ ИЗВЛЕЧЕНИЕ
echo ========================================
echo 📄 Используется: extract_frames.py
echo 🎯 Параметры: 30 кадров, равномерное распределение
echo 💾 Сохранение: локальная папка Images
echo.

python extract_frames.py

if %errorlevel% equ 0 (
    echo.
    echo ✅ Извлечение завершено успешно!
    echo 📁 Кадры сохранены в папке Images
    echo.
    echo 💡 Следующие шаги:
    echo    1. Проверьте кадры в папке Images
    echo    2. python ПОЛНЫЙ_PIPELINE.py --prepare-roboflow
    echo    3. Загрузите кадры в Roboflow для разметки
) else (
    echo ❌ Ошибки при извлечении кадров
    echo 💡 Проверьте логи выше
)
goto :end

:advanced_extract
echo.
echo 🚀 ПРОДВИНУТОЕ ИЗВЛЕЧЕНИЕ
echo ========================================
echo 📄 Используется: extract_30_frames.py
echo 🎯 Параметры: 30 кадров, равномерное распределение
echo 💾 Сохранение: облако UTair (авто-определение)
echo.

python extract_30_frames.py

if %errorlevel% equ 0 (
    echo.
    echo ✅ Извлечение завершено успешно!
    echo 🌐 Кадры сохранены в облаке UTair
    echo 🔗 Проверить: https://cloud.utair.ru/apps/files/?dir=/TST_project
    echo.
    echo 💡 Следующие шаги:
    echo    1. Проверьте кадры в облаке
    echo    2. Загрузите кадры в Roboflow для разметки
    echo    3. Обучите модель детекции
) else (
    echo ❌ Ошибки при извлечении кадров
    echo 💡 Проверьте логи в extract_30_frames.log
)
goto :end

:setup_cloud
echo.
echo 🔧 НАСТРОЙКА ОБЛАКА
echo ========================================
echo.

if exist НАСТРОИТЬ_ОБЛАКО.bat (
    call НАСТРОИТЬ_ОБЛАКО.bat
    echo.
    echo 🔄 После настройки облака запустите этот скрипт снова
) else (
    echo ❌ Файл НАСТРОИТЬ_ОБЛАКО.bat не найден
    echo 💡 Скачайте полный проект или создайте ссылку вручную
)
goto :end

:check_sync
echo.
echo 🔍 ПРОВЕРКА СИНХРОНИЗАЦИИ
echo ========================================
echo.

if exist check_cloud_sync.py (
    python check_cloud_sync.py --interactive
) else (
    echo 📊 Быстрая проверка:
    echo.
    
    if exist Videos\ (
        echo ✅ Папка Videos доступна
        
        set /a video_count=0
        for %%f in (Videos\*.mp4 Videos\*.avi Videos\*.mov) do set /a video_count+=1
        
        echo 📹 Видеофайлов: %video_count%
        
        if %video_count% GTR 0 (
            echo ✅ Видео готовы к обработке
        ) else (
            echo ⚠️ Видеофайлы не найдены
            echo 🌐 Проверьте: https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos
        )
    ) else (
        echo ❌ Папка Videos недоступна
        echo 🔧 Запустите: НАСТРОИТЬ_ОБЛАКО.bat
    )
)
goto :end

:invalid_choice
echo ❌ Неверный выбор. Попробуйте снова.
pause
goto :choice

:cancel
echo 👋 Операция отменена
goto :end

:end
echo.
echo ========================================
echo            СПРАВКА
echo ========================================
echo.
echo 📚 Документация:
echo    • СИНХРОНИЗАЦИЯ_UTAIR_CLOUD.md - работа с облаком
echo    • README_ROBOFLOW_PIPELINE.md - полный pipeline
echo    • БЫСТРЫЙ_СТАРТ_UTAIR_CLOUD.md - краткое руководство
echo.
echo 🛠️ Полезные команды:
echo    python extract_frames.py - быстрое извлечение
echo    python extract_30_frames.py - продвинутое извлечение
echo    python check_cloud_sync.py - проверка облака
echo    python ПОЛНЫЙ_PIPELINE.py --full - полный pipeline
echo.
echo 🌐 Ваше облако: https://cloud.utair.ru/apps/files/?dir=/TST_project
echo 💻 GitHub: https://github.com/EmilioSedici16/TST_git_repository
echo.
pause