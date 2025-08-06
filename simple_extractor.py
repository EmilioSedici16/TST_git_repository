#!/usr/bin/env python3
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
    
    # Получаем информацию о видео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📊 Видео: {total_frames} кадров, {fps:.1f} FPS")
    print(f"📁 Сохранение в: {output_dir}")
    
    frame_count = 0
    extracted_count = 0
    
    try:
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
                else:
                    print(f"   ❌ Ошибка сохранения {filename}")
                
                if max_frames and extracted_count >= max_frames:
                    print(f"   ⏹️ Достигнут лимит: {max_frames} кадров")
                    break
            
            frame_count += 1
            
            # Показываем прогресс
            if frame_count % 100 == 0:
                print(f"   📈 Обработано: {frame_count}/{total_frames}")
    
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    finally:
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
    
    successful_videos = 0
    total_images = 0
    
    for video_file in video_files:
        print(f"\n{'='*60}")
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
            successful_videos += 1
            
            # Подсчитываем изображения
            if output_dir.exists():
                image_count = len(list(output_dir.glob("*.jpg")))
                total_images += image_count
                print(f"   📸 Извлечено изображений: {image_count}")
        else:
            print(f"❌ {video_file.name} - ошибка")
    
    print(f"\n{'='*60}")
    print("🎉 Обработка завершена!")
    print(f"📊 Статистика:")
    print(f"   - Успешно обработано видео: {successful_videos}/{len(video_files)}")
    print(f"   - Всего извлечено изображений: {total_images}")

if __name__ == "__main__":
    print("🚀 Простой экстрактор кадров из видео")
    print("📋 Параметры:")
    print("   - Интервал кадров: каждый 60-й")
    print("   - Максимум кадров на видео: 5")
    print("   - Формат: JPEG")
    print()
    
    process_videos() 