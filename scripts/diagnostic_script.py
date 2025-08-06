#!/usr/bin/env python3
"""
Диагностический скрипт для проверки окружения Python
"""

import sys
import os
import subprocess
import platform

def check_python_installation():
    """Проверка установки Python"""
    print("🐍 Диагностика Python окружения")
    print("=" * 50)
    
    print(f"Python версия: {sys.version}")
    print(f"Python путь: {sys.executable}")
    print(f"Платформа: {platform.platform()}")
    print(f"Архитектура: {platform.architecture()}")
    
    print("\n📁 Python PATH:")
    for path in sys.path:
        print(f"  {path}")
    
    print(f"\n🔧 Рабочая директория: {os.getcwd()}")
    
    # Проверка pip
    try:
        import pip
        print(f"✅ pip версия: {pip.__version__}")
    except ImportError:
        print("❌ pip не найден")
    
    # Проверка основных библиотек
    libraries = ['cv2', 'numpy', 'ultralytics', 'streamlit', 'roboflow']
    
    print("\n📚 Проверка библиотек:")
    for lib in libraries:
        try:
            __import__(lib)
            print(f"  ✅ {lib}")
        except ImportError:
            print(f"  ❌ {lib} - НЕ УСТАНОВЛЕНА")

if __name__ == "__main__":
    check_python_installation()