@echo off
chcp 65001
echo ⚡ Быстрые команды для проекта
echo ================================
echo.

:menu
echo Выберите действие:
echo 1 - Проверить видео
echo 2 - Извлечь кадры из всех видео
echo 3 - Извлечь кадры из одного видео
echo 4 - Детекция на тестовом изображении
echo 5 - Детекция на видео (безопасность)
echo 6 - Запустить веб-интерфейс
echo 7 - Проверить установленные библиотеки
echo 8 - Интерактивная командная строка (Python)
echo 9 - Выход
echo.

set /p choice="Введите номер (1-9): "

if "%choice%"=="1" goto check_videos
if "%choice%"=="2" goto extract_all
if "%choice%"=="3" goto extract_one
if "%choice%"=="4" goto test_detection
if "%choice%"=="5" goto safety_detection
if "%choice%"=="6" goto web_interface
if "%choice%"=="7" goto check_libs
if "%choice%"=="8" goto interactive_cmd
if "%choice%"=="9" goto exit

echo Неверный выбор!
goto menu

:check_videos
echo 🎬 Проверка видеофайлов...
python -c "
import cv2
from pathlib import Path
videos = list(Path('Videos').glob('*.mp4'))
print(f'📊 Найдено видео: {len(videos)}')
for v in videos:
    cap = cv2.VideoCapture(str(v))
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'✅ {v.name}: {frames} кадров, {fps:.1f} FPS')
        cap.release()
    else:
        print(f'❌ {v.name}: Ошибка открытия')
"
pause
goto menu

:extract_all
echo 🖼️ Извлечение кадров из всех видео...
python video_to_images.py
pause
goto menu

:extract_one
echo 📹 Доступные видео:
dir Videos /b
echo.
set /p video_name="Введите имя видео: "
python video_to_images.py --single-video "Videos/%video_name%"
pause
goto menu

:test_detection
echo 🧪 Тест детекции на тестовом изображении...
python detect.py --source data/test_image.jpg --save --show
pause
goto menu

:safety_detection
echo 📹 Доступные видео:
dir Videos /b
echo.
set /p video_name="Введите имя видео: "
python safety_detection.py --source "Videos/%video_name%" --save
pause
goto menu

:web_interface
echo 🌐 Запуск веб-интерфейса...
echo Откроется в браузере на http://localhost:8501
echo Для остановки нажмите Ctrl+C
streamlit run app.py
pause
goto menu

:check_libs
echo 📚 Проверка библиотек...
python -c "
libs = ['cv2', 'numpy', 'ultralytics', 'streamlit', 'roboflow', 'PIL']
for lib in libs:
    try:
        __import__(lib)
        print(f'✅ {lib}')
    except ImportError:
        print(f'❌ {lib}')
"
pause
goto menu

:interactive_cmd
echo 🐍 Запуск интерактивной командной строки...
python cmd_helper.py
goto menu

:exit
echo 👋 До свидания!
exit