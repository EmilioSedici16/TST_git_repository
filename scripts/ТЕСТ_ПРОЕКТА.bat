@echo off
chcp 65001
echo 🧪 Тестирование основных компонентов проекта...
echo.

echo 📋 1. Проверка базового тестирования:
python test_basic.py
echo.

echo 📋 2. Проверка скрипта извлечения кадров:
python video_to_images.py --help
echo.

echo 📋 3. Проверка детекции:
python detect.py --help
echo.

echo 📋 4. Проверка безопасности:
python safety_detection.py --help
echo.

echo 📋 5. Проверка утилит:
python -c "from utils import RoboflowManager, ImageProcessor; print('✅ Утилиты загружены успешно')"
echo.

echo ✅ Тестирование завершено!
pause