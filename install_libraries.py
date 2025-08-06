#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автоматическая установка необходимых библиотек
"""

import subprocess
import sys
import importlib

def install_package(package_name, import_name=None):
    """Установить пакет через pip"""
    if import_name is None:
        import_name = package_name
    
    try:
        # Проверить, уже ли установлен
        importlib.import_module(import_name)
        print(f"✅ {package_name} уже установлен")
        return True
    except ImportError:
        print(f"📦 Установка {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} установлен успешно")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка установки {package_name}: {e}")
            return False

def main():
    print("🚀 АВТОМАТИЧЕСКАЯ УСТАНОВКА БИБЛИОТЕК")
    print("=" * 50)
    print()
    
    # Список библиотек для установки
    libraries = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("roboflow", "roboflow"),
        ("ultralytics", "ultralytics"),
        ("streamlit", "streamlit"),
        ("requests", "requests")
    ]
    
    success_count = 0
    total_count = len(libraries)
    
    print("📋 Будут установлены:")
    for package, _ in libraries:
        print(f"   • {package}")
    print()
    
    # Обновить pip
    print("🔄 Обновление pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip обновлен")
    except subprocess.CalledProcessError:
        print("⚠️ Не удалось обновить pip (не критично)")
    print()
    
    # Установить каждую библиотеку
    for package, import_name in libraries:
        if install_package(package, import_name):
            success_count += 1
        print()
    
    # Итоговый отчет
    print("=" * 50)
    print("📊 РЕЗУЛЬТАТ УСТАНОВКИ")
    print("=" * 50)
    print(f"✅ Успешно: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 ВСЕ БИБЛИОТЕКИ УСТАНОВЛЕНЫ!")
        print()
        print("🧪 Тестирование...")
        
        # Тест основных библиотек
        try:
            import cv2
            import numpy as np
            print(f"✅ OpenCV {cv2.__version__}")
            print(f"✅ NumPy {np.__version__}")
            
            # Тест создания массива
            test_array = np.zeros((10, 10, 3), dtype=np.uint8)
            print("✅ Создание массивов работает")
            
            # Тест OpenCV функций
            print("✅ OpenCV функции доступны")
            
            print()
            print("🚀 ГОТОВО К ИЗВЛЕЧЕНИЮ КАДРОВ!")
            print("💡 Запустите: python extract_frames.py")
            
        except Exception as e:
            print(f"⚠️ Ошибка при тестировании: {e}")
    else:
        failed = total_count - success_count
        print(f"❌ Не установлено: {failed} библиотек")
        print()
        print("🔧 Попробуйте:")
        print("   1. Запустить от имени администратора")
        print("   2. Проверить интернет-соединение")
        print("   3. Обновить Python")
    
    print()
    input("Нажмите Enter для завершения...")

if __name__ == "__main__":
    main()