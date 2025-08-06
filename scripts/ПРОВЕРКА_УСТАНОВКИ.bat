@echo off
chcp 65001
echo 🔍 Проверка установленных зависимостей...
echo.

echo 📋 Версия Python:
python --version
echo.

echo 📋 Проверка pip:
python -m pip --version
echo.

echo 🧪 Тест основных библиотек:
python -c "
import sys
print('✅ Python:', sys.version_info)

libraries = ['cv2', 'numpy', 'ultralytics', 'streamlit', 'roboflow', 'PIL', 'requests']
for lib in libraries:
    try:
        __import__(lib)
        print(f'✅ {lib} - УСТАНОВЛЕНА')
    except ImportError:
        print(f'❌ {lib} - НЕ НАЙДЕНА')
"
echo.

echo 📊 Список установленных пакетов:
python -m pip list | findstr /i "opencv numpy ultralytics streamlit roboflow pillow requests"
echo.

echo ✅ Проверка завершена!
pause