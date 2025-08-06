#!/usr/bin/env python3
"""
Скрипт для извлечения кадров из видео и создания изображений
Соответствует структуре папок в Images/
"""

import cv2
import os
import argparse
from pathlib import Path
import time

def extract_frames_from_video(video_path, output_dir, frame_interval=30, max_frames=None):
    """
    Извлекает кадры из видео с заданным интервалом
    
    Args:
        video_path: Путь к видеофайлу
        output_dir: Папка для сохранения изображений
        frame_interval: Интервал между кадрами (каждый N-й кадр)
        max_frames: Максимальное количество кадров для извлечения
    """
    print(f"🎬 Обработка видео: {video_path}")
    
    # Создаем папку если её нет
    os.makedirs(output_dir, exist_ok=True)
    
    # Открываем видео с дополнительными параметрами для лучшей совместимости
    cap = cv2.VideoCapture(video_path)
    
    # Устанавливаем дополнительные параметры для лучшей совместимости
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео: {video_path}")
        return False
    
    # Получаем информацию о видео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 Информация о видео:")
    print(f"   - Всего кадров: {total_frames}")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - Длительность: {duration:.2f} секунд")
    print(f"   - Интервал извлечения: каждый {frame_interval}-й кадр")
    
    frame_count = 0
    extracted_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 10
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"   ⚠️ Слишком много последовательных ошибок чтения кадров")
                    break
                continue
            
            consecutive_failures = 0  # Сбрасываем счетчик при успешном чтении
            
            # Извлекаем кадр с заданным интервалом
            if frame_count % frame_interval == 0:
                # Проверяем, что кадр не пустой
                if frame is not None and frame.size > 0:
                    # Формируем имя файла
                    timestamp = frame_count / fps if fps > 0 else frame_count
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    
                    filename = f"frame_{frame_count:06d}_{minutes:02d}m_{seconds:02d}s.jpg"
                    output_path = os.path.join(output_dir, filename)
                    
                    try:
                        # Сохраняем кадр с дополнительными параметрами
                        success = cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if success:
                            extracted_count += 1
                            print(f"   ✅ Сохранен кадр: {filename}")
                        else:
                            print(f"   ❌ Ошибка сохранения: {filename}")
                    except Exception as e:
                        print(f"   ❌ Исключение при сохранении {filename}: {e}")
                
                # Проверяем лимит кадров
                if max_frames and extracted_count >= max_frames:
                    print(f"   ⏹️ Достигнут лимит кадров: {max_frames}")
                    break
            
            frame_count += 1
            
            # Показываем прогресс каждые 100 кадров
            if frame_count % 100 == 0:
                print(f"   📈 Обработано кадров: {frame_count}/{total_frames}")
    
    except Exception as e:
        print(f"   ❌ Ошибка при обработке видео: {e}")
    
    finally:
        cap.release()
    
    print(f"✅ Извлечение завершено!")
    print(f"   - Обработано кадров: {frame_count}")
    print(f"   - Извлечено изображений: {extracted_count}")
    print(f"   - Сохранено в: {output_dir}")
    
    return extracted_count > 0

def process_all_videos(videos_dir="Videos", images_dir="Images", frame_interval=30, max_frames=50):
    """
    Обрабатывает все видео в папке Videos и создает изображения в Images
    
    Args:
        videos_dir: Папка с видеофайлами
        images_dir: Папка для сохранения изображений
        frame_interval: Интервал между кадрами
        max_frames: Максимальное количество кадров на видео
    """
    print("🚀 Начинаем обработку всех видео...")
    
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
        success = extract_frames_from_video(
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

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Извлечение кадров из видео')
    
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
    
    print("🎬 Скрипт извлечения кадров из видео")
    print(f"📁 Папка с видео: {args.videos_dir}")
    print(f"📁 Папка для изображений: {args.images_dir}")
    print(f"⏱️ Интервал кадров: {args.frame_interval}")
    print(f"📊 Максимум кадров на видео: {args.max_frames}")
    
    if args.single_video:
        # Обрабатываем одно видео
        video_path = Path(args.single_video)
        if not video_path.exists():
            print(f"❌ Видеофайл не найден: {args.single_video}")
            return
        
        folder_name = video_path.stem
        output_dir = Path(args.images_dir) / folder_name
        
        extract_frames_from_video(
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