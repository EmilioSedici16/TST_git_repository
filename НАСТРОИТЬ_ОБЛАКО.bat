@echo off
chcp 65001 >nul
echo ========================================
echo   Настройка синхронизации UTair Cloud
echo ========================================
echo.

echo 🌐 Ваши видео в облаке:
echo https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos
echo.

REM Проверка текущего состояния папки Videos
if exist Videos\ (
    echo 📁 Папка Videos уже существует
    
    REM Проверка, является ли она ссылкой
    dir Videos | find "SYMLINK" >nul
    if %errorlevel% equ 0 (
        echo ✅ Videos уже настроена как ссылка на облако
        goto :check_access
    ) else (
        echo ⚠️ Videos - обычная папка (не ссылка на облако)
        echo.
        set /p backup="Создать резервную копию локальных файлов? (y/n): "
        if /i "%backup%"=="y" (
            if not exist Videos_backup mkdir Videos_backup
            copy Videos\*.* Videos_backup\ 2>nul
            echo 💾 Локальные файлы скопированы в Videos_backup\
        )
        
        echo 🗑️ Удаление локальной папки Videos...
        rmdir Videos /s /q 2>nul
    )
) else (
    echo 📂 Папка Videos не существует
)

echo.
echo ========================================
echo      Варианты подключения облака
echo ========================================
echo.
echo 1. WebDAV (рекомендуемый) - подключить как сетевой диск
echo 2. Символическая ссылка на уже подключенное облако
echo 3. Ручная синхронизация через веб-интерфейс
echo 4. Настроить позже
echo.

set /p choice="Выберите вариант (1-4): "

if "%choice%"=="1" goto :webdav_setup
if "%choice%"=="2" goto :symlink_setup
if "%choice%"=="3" goto :manual_setup
if "%choice%"=="4" goto :skip_setup
goto :invalid_choice

:webdav_setup
echo.
echo 🌐 НАСТРОЙКА WEBDAV
echo ========================================
echo.

echo 📋 Для подключения WebDAV нужны:
echo    • Логин UTair Cloud
echo    • Пароль UTair Cloud
echo.

set /p username="Введите логин UTair Cloud: "
if "%username%"=="" (
    echo ❌ Логин не может быть пустым
    goto :webdav_setup
)

echo.
echo 🔗 Подключение UTair Cloud как сетевой диск...
echo.

REM Поиск свободной буквы диска
set drive_letter=
for %%d in (Z Y X W V U T S R Q P O N M L K) do (
    if not exist %%d:\ (
        set drive_letter=%%d
        goto :found_drive
    )
)

:found_drive
if "%drive_letter%"=="" (
    echo ❌ Не найдено свободных букв дисков
    echo 💡 Освободите один из дисков и повторите попытку
    goto :manual_setup
)

echo 💾 Используем диск %drive_letter%: для UTair Cloud
echo.

REM Подключение WebDAV
echo 🔄 Подключение...
net use %drive_letter%: https://cloud.utair.ru/remote.php/dav/files/%username%/ /persistent:yes

if %errorlevel% equ 0 (
    echo ✅ UTair Cloud подключен как диск %drive_letter%:
    
    REM Проверка существования папки TST_project/Videos
    if exist %drive_letter%:\TST_project\Videos\ (
        echo ✅ Папка TST_project/Videos найдена в облаке
        
        REM Создание символической ссылки
        echo 🔗 Создание ссылки Videos → %drive_letter%:\TST_project\Videos
        mklink /D Videos "%drive_letter%:\TST_project\Videos"
        
        if %errorlevel% equ 0 (
            echo ✅ Ссылка создана успешно!
            goto :success_setup
        ) else (
            echo ❌ Ошибка создания ссылки
            goto :manual_setup
        )
    ) else (
        echo ❌ Папка TST_project/Videos не найдена в облаке
        echo 📁 Создание папки в облаке...
        
        mkdir "%drive_letter%:\TST_project" 2>nul
        mkdir "%drive_letter%:\TST_project\Videos" 2>nul
        
        if exist %drive_letter%:\TST_project\Videos\ (
            echo ✅ Папка создана в облаке
            mklink /D Videos "%drive_letter%:\TST_project\Videos"
            goto :success_setup
        ) else (
            echo ❌ Не удалось создать папку в облаке
            goto :manual_setup
        )
    )
) else (
    echo ❌ Ошибка подключения WebDAV
    echo 💡 Возможные причины:
    echo    • Неверный логин/пароль
    echo    • Проблемы с сетью
    echo    • UTair Cloud недоступен
    echo.
    goto :manual_setup
)

:symlink_setup
echo.
echo 🔗 СОЗДАНИЕ ССЫЛКИ НА ОБЛАКО
echo ========================================
echo.

echo 📋 Для создания ссылки введите путь к уже подключенному облаку:
echo Примеры:
echo    Z:\TST_project\Videos
echo    \\server\share\TST_project\Videos
echo    C:\Users\%USERNAME%\UTairCloud\TST_project\Videos
echo.

set /p cloud_path="Введите полный путь к Videos в облаке: "

if "%cloud_path%"=="" (
    echo ❌ Путь не может быть пустым
    goto :symlink_setup
)

if exist "%cloud_path%" (
    echo ✅ Папка найдена: %cloud_path%
    echo 🔗 Создание ссылки...
    
    mklink /D Videos "%cloud_path%"
    
    if %errorlevel% equ 0 (
        echo ✅ Ссылка создана успешно!
        goto :success_setup
    ) else (
        echo ❌ Ошибка создания ссылки
        goto :manual_setup
    )
) else (
    echo ❌ Папка не найдена: %cloud_path%
    echo 💡 Проверьте путь и доступность облака
    goto :symlink_setup
)

:manual_setup
echo.
echo 📋 РУЧНАЯ СИНХРОНИЗАЦИЯ
echo ========================================
echo.

echo 🔧 Настройка ручной синхронизации:
echo.
echo 1. Создаем локальную папку Videos
mkdir Videos 2>nul

echo 2. Создаем инструкцию по синхронизации
echo. > Videos\README_СИНХРОНИЗАЦИЯ.txt
echo РУЧНАЯ СИНХРОНИЗАЦИЯ С УТАИР ОБЛАКО >> Videos\README_СИНХРОНИЗАЦИЯ.txt
echo. >> Videos\README_СИНХРОНИЗАЦИЯ.txt
echo Ваши видео в облаке: >> Videos\README_СИНХРОНИЗАЦИЯ.txt
echo https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos >> Videos\README_СИНХРОНИЗАЦИЯ.txt
echo. >> Videos\README_СИНХРОНИЗАЦИЯ.txt
echo ПЕРЕД ОБРАБОТКОЙ: >> Videos\README_СИНХРОНИЗАЦИЯ.txt
echo 1. Перейдите по ссылке выше >> Videos\README_СИНХРОНИЗАЦИЯ.txt
echo 2. Скачайте нужные видео в эту папку >> Videos\README_СИНХРОНИЗАЦИЯ.txt
echo. >> Videos\README_СИНХРОНИЗАЦИЯ.txt
echo ПОСЛЕ ОБРАБОТКИ: >> Videos\README_СИНХРОНИЗАЦИЯ.txt
echo 1. Загрузите новые видео в облако >> Videos\README_СИНХРОНИЗАЦИЯ.txt
echo 2. Удалите локальные копии для экономии места >> Videos\README_СИНХРОНИЗАЦИЯ.txt

echo ✅ Создана папка Videos с инструкцией по синхронизации
echo 📄 См. файл: Videos\README_СИНХРОНИЗАЦИЯ.txt
echo.
echo 🌐 Ваше облако: https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos
goto :end_setup

:skip_setup
echo.
echo ⏭️ Настройка облака пропущена
echo 💡 Вы можете настроить позже, запустив этот скрипт снова
goto :end_setup

:success_setup
echo.
echo ========================================
echo        НАСТРОЙКА ЗАВЕРШЕНА!
echo ========================================
echo.

:check_access
echo 🔍 Проверка доступа к видео...

if exist Videos\ (
    REM Подсчет видеофайлов
    set /a count=0
    for %%f in (Videos\*.mp4 Videos\*.avi Videos\*.mov Videos\*.mkv) do set /a count+=1
    
    echo 📊 Найдено видеофайлов: %count%
    
    if %count% GTR 0 (
        echo.
        echo 📹 Доступные видео:
        for %%f in (Videos\*.mp4 Videos\*.avi Videos\*.mov Videos\*.mkv) do echo    %%~nxf
        echo.
        echo ✅ Видео готовы к обработке!
        echo.
        set /p extract="Запустить извлечение кадров сейчас? (y/n): "
        if /i "%extract%"=="y" (
            echo 🎬 Запуск извлечения кадров...
            python extract_frames.py
            if %errorlevel% equ 0 (
                echo ✅ Кадры извлечены успешно!
            ) else (
                echo ⚠️ Возможны ошибки при извлечении
            )
        )
    ) else (
        echo ⚠️ Видеофайлы не найдены
        echo.
        echo 💡 Возможные причины:
        echo    • Видео еще не загружены в облако
        echo    • Проблемы с синхронизацией
        echo    • Неправильный путь к облаку
        echo.
        echo 🌐 Проверьте облако: https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos
    )
) else (
    echo ❌ Папка Videos недоступна
)

goto :end_setup

:invalid_choice
echo ❌ Неверный выбор. Попробуйте снова.
goto :choice

:end_setup
echo.
echo ========================================
echo            ИНФОРМАЦИЯ
echo ========================================
echo.
echo 🌐 Облако UTair: https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos
echo 📁 Локальная папка: Videos\
echo 📚 Документация: СИНХРОНИЗАЦИЯ_UTAIR_CLOUD.md
echo.
echo 🔄 Рабочий процесс:
echo    1. Загрузить/скачать видео через облако
echo    2. python extract_frames.py (извлечь кадры)
echo    3. python ПОЛНЫЙ_PIPELINE.py --prepare-roboflow
echo    4. Работа с Roboflow (разметка + обучение)
echo    5. python detect.py (детекция на новых данных)
echo.
echo 💡 Для повторной настройки запустите этот скрипт снова
echo.
pause