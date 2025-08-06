#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой тест для проверки работы Python и библиотек
"""

import sys
import os
from pathlib import Path

def test_python():
    """Проверить Python и библиотеки"""
    print("🐍 ТЕСТ PYTHON")
    print("=" * 40)
    print(f"Версия Python: {sys.version}")
    print(f"Исполняемый файл: {sys.executable}")
    print(f"Текущая папка: {os.getcwd()}")
    print(f"PATH включает: {sys.executable in os.environ.get('PATH', '')}")
    print()
    
    # Проверка библиотек
    libraries = ['cv2', 'numpy', 'pathlib']
    
    print("📦 ПРОВЕРКА БИБЛИОТЕК:")
    print("-" * 40)
    
    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'неизвестно')
            print(f"✅ {lib}: {version}")
        except ImportError as e:
            print(f"❌ {lib}: НЕ УСТАНОВЛЕН ({e})")
    
    print()
    
    # Проверка видео файлов
    print("📹 ПРОВЕРКА ВИДЕО:")
    print("-" * 40)
    
    videos_dir = Path("Videos")
    if videos_dir.exists():
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(videos_dir.glob(f"*{ext}"))
            video_files.extend(videos_dir.glob(f"*{ext.upper()}"))
        
        print(f"📁 Папка Videos: НАЙДЕНА")
        print(f"📹 Видеофайлов: {len(video_files)}")
        
        for i, video in enumerate(video_files[:5], 1):
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"   {i}. {video.name} ({size_mb:.1f} MB)")
        
        if len(video_files) > 5:
            print(f"   ... и ещё {len(video_files) - 5} файлов")
    else:
        print(f"❌ Папка Videos: НЕ НАЙДЕНА")
        print("💡 Запустите НАСТРОИТЬ_ОБЛАКО.bat")
    
    print()
    print("🎯 ГОТОВНОСТЬ К ИЗВЛЕЧЕНИЮ КАДРОВ:")
    print("-" * 40)
    
    # Итоговая проверка
    python_ok = True
    opencv_ok = False
    videos_ok = videos_dir.exists() and len(list(videos_dir.glob("*.mp4"))) > 0
    
    try:
        import cv2
        opencv_ok = True
    except ImportError:
        pass
    
    if python_ok and opencv_ok and videos_ok:
        print("✅ ВСЁ ГОТОВО для извлечения кадров!")
        print("🚀 Запустите: python extract_frames.py")
    else:
        print("⚠️ Требуется настройка:")
        if not opencv_ok:
            print("   📦 pip install opencv-python")
        if not videos_ok:
            print("   📁 Настройте доступ к видео")
    
    return python_ok, opencv_ok, videos_ok

def test_frame_extraction():
    """Быстрый тест извлечения кадров"""
    try:
        import cv2
        videos_dir = Path("Videos")
        
        if not videos_dir.exists():
            print("❌ Папка Videos недоступна")
            return False
        
        video_files = list(videos_dir.glob("*.mp4"))
        if not video_files:
            print("❌ MP4 файлы не найдены")
            return False
        
        test_video = video_files[0]
        print(f"🎬 Тестируем: {test_video.name}")
        
        cap = cv2.VideoCapture(str(test_video))
        if not cap.isOpened():
            print("❌ Не удалось открыть видео")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Видео открыто: {total_frames} кадров, {fps:.1f} FPS")
        
        # Попробовать прочитать первый кадр
        ret, frame = cap.read()
        if ret:
            print(f"✅ Кадр прочитан: {frame.shape}")
        else:
            print("❌ Не удалось прочитать кадр")
            return False
        
        cap.release()
        print("✅ Тест извлечения УСПЕШЕН")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
        return False

if __name__ == "__main__":
    print("🔍 ДИАГНОСТИКА СИСТЕМЫ ДЛЯ ИЗВЛЕЧЕНИЯ КАДРОВ")
    print("=" * 60)
    print()
    
    # Основной тест
    python_ok, opencv_ok, videos_ok = test_python()
    
    # Тест извлечения если всё готово
    if python_ok and opencv_ok and videos_ok:
        print()
        print("🧪 ТЕСТ ИЗВЛЕЧЕНИЯ КАДРОВ:")
        print("-" * 40)
        extraction_ok = test_frame_extraction()
        
        if extraction_ok:
            print()
            print("🎉 СИСТЕМА ПОЛНОСТЬЮ ГОТОВА!")
            print("💡 Запустите ИЗВЛЕЧЬ_30_КАДРОВ.bat или python extract_frames.py")
    
    print()
    input("Нажмите Enter для завершения...")