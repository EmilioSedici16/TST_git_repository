#!/usr/bin/env python3
"""
Установка зависимостей для проекта компьютерного зрения
"""

import subprocess
import sys
import os

def install_package(package_name):
    """Устанавливает пакет через pip"""
    print(f"📦 Устанавливаем {package_name}...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', package_name
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {package_name} установлен успешно")
            return True
        else:
            print(f"❌ Ошибка установки {package_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Исключение при установке {package_name}: {e}")
        return False

def main():
    """Основная функция установки"""
    print("🚀 Установка зависимостей для проекта компьютерного зрения")
    print("=" * 60)
    
    # Список необходимых пакетов
    packages = [
        'opencv-python',      # Обработка изображений и видео
        'numpy',              # Численные вычисления
        'ultralytics',        # YOLOv8 модели
        'streamlit',          # Веб-интерфейс
        'roboflow',           # API для Roboflow
        'pillow',             # Обработка изображений
        'matplotlib',         # Визуализация
        'requests'            # HTTP запросы
    ]
    
    print("📋 Пакеты для установки:")
    for package in packages:
        print(f"   - {package}")
    
    print(f"\n🔧 Начинаем установку...")
    
    successful_installs = 0
    failed_installs = []
    
    for package in packages:
        if install_package(package):
            successful_installs += 1
        else:
            failed_installs.append(package)
        print()  # Пустая строка для разделения
    
    # Результаты
    print("=" * 60)
    print("📊 Результаты установки:")
    print(f"✅ Успешно установлено: {successful_installs}/{len(packages)}")
    
    if failed_installs:
        print(f"❌ Не удалось установить: {', '.join(failed_installs)}")
        print("\n💡 Рекомендации:")
        print("   - Попробуйте обновить pip: python -m pip install --upgrade pip")
        print("   - Проверьте подключение к интернету")
        print("   - Попробуйте установить пакеты по одному")
    else:
        print("🎉 Все пакеты установлены успешно!")
    
    print(f"\n🚀 Проект готов к работе!")

if __name__ == "__main__":
    main() 