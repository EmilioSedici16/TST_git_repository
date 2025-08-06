#!/usr/bin/env python3
"""
Альтернативный скрипт для извлечения кадров из видео с использованием ffmpeg
"""

import os
import argparse
import subprocess
from pathlib import Path
import time

def extract_frames_with_ffmpeg(video_path, output_dir, frame_interval=30, max_frames=None):
    """
    Извлекает кадры из видео с помощью ffmpeg
    
    Args:
        video_path: Путь к видеофайлу
        output_dir: Папка для сохранения изображений
        frame_interval: Интервал между кадрами (каждый N-й кадр)
        max_frames: Максимальное количество кадров для извлечения
    """
    print(f"🎬 Обработка видео: {video_path}")
    
    # Создаем папку если её нет
    os.makedirs(output_dir, exist_ok=True)
    
    # Формируем команду ffmpeg
    # -vf "select=not(mod(n,{frame_interval}))" - выбираем каждый N-й кадр
    # -vsync 0 - отключаем синхронизацию
    # -frame_pts 1 - добавляем временные метки к именам файлов
    
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', f'select=not(mod(n,{frame_interval}))',
        '-vsync', '0',
        '-frame_pts', '1',
        '-q:v', '2',  # Качество JPEG (2 = высокое качество)
        os.path.join(output_dir, 'frame_%06d_%s.jpg')
    ]
    
    print(f"🔧 Команда ffmpeg: {' '.join(cmd)}")
    
    try:
        # Запускаем ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ FFmpeg выполнен успешно")
            
            # Подсчитываем количество созданных файлов
            image_files = list(Path(output_dir).glob("*.jpg"))
            print(f"📸 Создано изображений: {len(image_files)}")
            
            # Если указан лимит кадров, удаляем лишние
            if max_frames and len(image_files) > max_frames:
                # Сортируем файлы по имени и оставляем только первые max_frames
                sorted_files = sorted(image_files, key=lambda x: x.name)
                for file_to_delete in sorted_files[max_frames:]:
                    file_to_delete.unlink()
                print(f"🗑️ Удалено лишних файлов: {len(image_files) - max_frames}")
                image_files = sorted_files[:max_frames]
            
            return len(image_files) > 0
        else:
            print(f"❌ Ошибка ffmpeg: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Превышено время выполнения ffmpeg")
        return False
    except Exception as e:
        print(f"❌ Исключение при выполнении ffmpeg: {e}")
        return False

def process_all_videos(videos_dir="Videos", images_dir="Images", frame_interval=30, max_frames=50):
    """
    Обрабатывает все видео в папке Videos и создает изображения в Images
    
    Args:
        videos_dir: Папка с видеофайлами
        images_dir: Папка для сохранения изображений
        frame_interval: Интервал между кадрами
        max_frames: Максимальное количество кадров на видео
    """
    print("🚀 Начинаем обработку всех видео с ffmpeg...")
    
    videos_path = Path(videos_dir)
    if not videos_path.exists():
        print(f"❌ Папка с видео не найдена: {videos_dir}")
        return
    
    # Получаем список всех видеофайлов
    video_files = list(videos_path.glob("*.mp4"))
    if not video_files:
        print(f"❌ Видеофайлы не найдены в папке: {videos_dir}")
        return
    
    print(f"📁 Найдено видеофайлов: {len(video_files)}")
    
    successful_videos = 0
    total_extracted = 0
    
    for video_file in video_files:
        print(f"\n{'='*60}")
        
        # Создаем имя папки для изображений (без расширения)
        folder_name = video_file.stem
        output_dir = Path(images_dir) / folder_name
        
        # Извлекаем кадры
        success = extract_frames_with_ffmpeg(
            str(video_file),
            str(output_dir),
            frame_interval=frame_interval,
            max_frames=max_frames
        )
        
        if success:
            print(f"✅ Видео обработано успешно: {video_file.name}")
            successful_videos += 1
            
            # Подсчитываем количество извлеченных изображений
            if output_dir.exists():
                image_count = len(list(output_dir.glob("*.jpg")))
                total_extracted += image_count
                print(f"   📸 Извлечено изображений: {image_count}")
        else:
            print(f"❌ Ошибка обработки видео: {video_file.name}")
    
    print(f"\n{'='*60}")
    print("🎉 Обработка всех видео завершена!")
    print(f"📊 Статистика:")
    print(f"   - Успешно обработано видео: {successful_videos}/{len(video_files)}")
    print(f"   - Всего извлечено изображений: {total_extracted}")

def check_ffmpeg():
    """Проверяет доступность ffmpeg"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg доступен")
            return True
        else:
            print("❌ FFmpeg не найден или не работает")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg не установлен")
        print("💡 Установите ffmpeg:")
        print("   Windows: https://ffmpeg.org/download.html")
        print("   или используйте: winget install ffmpeg")
        return False

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Извлечение кадров из видео с ffmpeg')
    
    parser.add_argument('--videos-dir', type=str, default='Videos',
                       help='Папка с видеофайлами')
    
    parser.add_argument('--images-dir', type=str, default='Images',
                       help='Папка для сохранения изображений')
    
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='Интервал между извлекаемыми кадрами (каждый N-й кадр)')
    
    parser.add_argument('--max-frames', type=int, default=50,
                       help='Максимальное количество кадров для извлечения из каждого видео')
    
    parser.add_argument('--single-video', type=str,
                       help='Обработать только одно видео (путь к файлу)')
    
    args = parser.parse_args()
    
    print("🎬 Скрипт извлечения кадров из видео с FFmpeg")
    print(f"📁 Папка с видео: {args.videos_dir}")
    print(f"📁 Папка для изображений: {args.images_dir}")
    print(f"⏱️ Интервал кадров: {args.frame_interval}")
    print(f"📊 Максимум кадров на видео: {args.max_frames}")
    
    # Проверяем доступность ffmpeg
    if not check_ffmpeg():
        return
    
    if args.single_video:
        # Обрабатываем одно видео
        video_path = Path(args.single_video)
        if not video_path.exists():
            print(f"❌ Видеофайл не найден: {args.single_video}")
            return
        
        folder_name = video_path.stem
        output_dir = Path(args.images_dir) / folder_name
        
        extract_frames_with_ffmpeg(
            str(video_path),
            str(output_dir),
            frame_interval=args.frame_interval,
            max_frames=args.max_frames
        )
    else:
        # Обрабатываем все видео
        process_all_videos(
            videos_dir=args.videos_dir,
            images_dir=args.images_dir,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames
        )

if __name__ == "__main__":
    main() 