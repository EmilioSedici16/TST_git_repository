#!/usr/bin/env python3
"""
Простой тест окружения
"""

import sys
import os
from pathlib import Path

def test_basic():
    """Базовые тесты"""
    print("🔍 Тест окружения Windows 11 + Python 3.13")
    print("=" * 50)
    
    # Python
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Путь: {sys.executable}")
    
    # Папки
    videos_dir = Path("Videos")
    images_dir = Path("Images")
    
    print(f"\n📁 Папка Videos: {'✅' if videos_dir.exists() else '❌'}")
    if videos_dir.exists():
        video_files = list(videos_dir.glob("*.mp4"))
        print(f"   Видеофайлов: {len(video_files)}")
        for vf in video_files:
            print(f"   - {vf.name}")
    
    print(f"📁 Папка Images: {'✅' if images_dir.exists() else '❌'}")
    
    # Зависимости
    print(f"\n📦 Зависимости:")
    
    try:
        import cv2
        print(f"   OpenCV: ✅ {cv2.__version__}")
    except ImportError:
        print("   OpenCV: ❌ не установлен")
    
    try:
        import numpy as np
        print(f"   NumPy: ✅ {np.__version__}")
    except ImportError:
        print("   NumPy: ❌ не установлен")
    
    try:
        import ultralytics
        print(f"   Ultralytics: ✅ {ultralytics.__version__}")
    except ImportError:
        print("   Ultralytics: ❌ не установлен")
    
    try:
        import streamlit
        print(f"   Streamlit: ✅ {streamlit.__version__}")
    except ImportError:
        print("   Streamlit: ❌ не установлен")
    
    print(f"\n✅ Тест завершен!")

if __name__ == "__main__":
    test_basic() 