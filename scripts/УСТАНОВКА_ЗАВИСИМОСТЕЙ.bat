@echo off
chcp 65001
echo 🐍 Установка зависимостей Python для проекта компьютерного зрения...
echo.

echo 📋 Проверка Python:
python --version
echo.

echo 📋 Проверка pip:
python -m pip --version
echo.

echo 📦 Установка зависимостей из requirements.txt:
python -m pip install -r requirements.txt
echo.

echo 🧪 Тест базовых библиотек:
python test_python_simple.py
echo.

echo ✅ Установка завершена!
pause