#!/usr/bin/env python3
"""
Скрипт для проверки и настройки окружения Windows
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python():
    """Проверяет версию Python"""
    print(f"🐍 Python версия: {sys.version}")
    print(f"📁 Python путь: {sys.executable}")
    return True

def check_pip():
    """Проверяет pip"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"📦 Pip доступен: {result.stdout.strip()}")
            return True
        else:
            print("❌ Pip не найден")
            return False
    except Exception as e:
        print(f"❌ Ошибка проверки pip: {e}")
        return False

def check_ffmpeg():
    """Проверяет FFmpeg"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ FFmpeg установлен")
            return True
        else:
            print("❌ FFmpeg не найден")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg не установлен")
        return False
    except Exception as e:
        print(f"❌ Ошибка проверки FFmpeg: {e}")
        return False

def install_ffmpeg():
    """Устанавливает FFmpeg через winget"""
    print("🔧 Устанавливаем FFmpeg...")
    try:
        # Проверяем доступность winget
        result = subprocess.run(['winget', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Winget доступен")
            
            # Устанавливаем FFmpeg
            install_result = subprocess.run(['winget', 'install', 'ffmpeg'], 
                                         capture_output=True, text=True)
            if install_result.returncode == 0:
                print("✅ FFmpeg установлен успешно")
                return True
            else:
                print(f"❌ Ошибка установки FFmpeg: {install_result.stderr}")
                return False
        else:
            print("❌ Winget не доступен")
            return False
    except FileNotFoundError:
        print("❌ Winget не найден")
        return False
    except Exception as e:
        print(f"❌ Ошибка установки: {e}")
        return False

def check_dependencies():
    """Проверяет Python зависимости"""
    required_packages = [
        'opencv-python',
        'numpy',
        'ultralytics',
        'streamlit',
        'roboflow'
    ]
    
    print("📋 Проверяем зависимости...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - не установлен")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Устанавливаем недостающие пакеты: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                result = subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                     capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ {package} установлен")
                else:
                    print(f"❌ Ошибка установки {package}: {result.stderr}")
            except Exception as e:
                print(f"❌ Ошибка установки {package}: {e}")
    else:
        print("✅ Все зависимости установлены")

def create_simple_extractor():
    """Создает простой скрипт для извлечения кадров без FFmpeg"""
    print("🔧 Создаем простой экстрактор кадров...")
    
    simple_extractor = '''#!/usr/bin/env python3
"""
Простой скрипт для извлечения кадров из видео
Использует только OpenCV без FFmpeg
"""

import cv2
import os
from pathlib import Path

def extract_frames_simple(video_path, output_dir, frame_interval=30, max_frames=10):
    """Простое извлечение кадров"""
    print(f"🎬 Обработка: {video_path}")
    
    # Создаем папку
    os.makedirs(output_dir, exist_ok=True)
    
    # Открываем видео
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Не удалось открыть: {video_path}")
        return False
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            filename = f"frame_{frame_count:06d}.jpg"
            output_path = os.path.join(output_dir, filename)
            
            if cv2.imwrite(output_path, frame):
                extracted_count += 1
                print(f"   ✅ {filename}")
            
            if max_frames and extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"✅ Извлечено: {extracted_count} кадров")
    return extracted_count > 0

def process_videos():
    """Обрабатывает все видео"""
    videos_dir = Path("Videos")
    images_dir = Path("Images")
    
    if not videos_dir.exists():
        print("❌ Папка Videos не найдена")
        return
    
    video_files = list(videos_dir.glob("*.mp4"))
    if not video_files:
        print("❌ Видеофайлы не найдены")
        return
    
    print(f"📁 Найдено видео: {len(video_files)}")
    
    for video_file in video_files:
        print(f"\\n{'='*50}")
        folder_name = video_file.stem
        output_dir = images_dir / folder_name
        
        success = extract_frames_simple(
            video_file, 
            output_dir, 
            frame_interval=60,  # Каждый 60-й кадр
            max_frames=5        # Максимум 5 кадров
        )
        
        if success:
            print(f"✅ {video_file.name} - успешно")
        else:
            print(f"❌ {video_file.name} - ошибка")

if __name__ == "__main__":
    process_videos()
'''
    
    with open('simple_extractor.py', 'w', encoding='utf-8') as f:
        f.write(simple_extractor)
    
    print("✅ Создан simple_extractor.py")

def main():
    """Основная функция"""
    print("🔍 Проверка окружения Windows 11 + Python 3.13")
    print("=" * 50)
    
    # Проверяем Python
    check_python()
    print()
    
    # Проверяем pip
    check_pip()
    print()
    
    # Проверяем зависимости
    check_dependencies()
    print()
    
    # Проверяем FFmpeg
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        print("\n🔧 FFmpeg не найден. Попробуем установить...")
        install_ffmpeg()
        print("\n🔄 Проверяем FFmpeg после установки...")
        check_ffmpeg()
    
    print("\n" + "=" * 50)
    print("📋 Рекомендации:")
    
    if ffmpeg_available or check_ffmpeg():
        print("✅ Можно использовать video_to_images_ffmpeg.py")
    else:
        print("⚠️ FFmpeg недоступен, создаем простой экстрактор")
        create_simple_extractor()
        print("✅ Используйте simple_extractor.py")
    
    print("\n🚀 Готово к работе!")

if __name__ == "__main__":
    main() 