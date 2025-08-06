import cv2
import os
import glob
from pathlib import Path

def extract_frames_from_video(video_path, output_dir, num_frames=30):
    """
    Извлекает указанное количество кадров из видео файла с равномерным распределением
    По умолчанию извлекает 30 кадров (первый кадр из каждого из 30 равных сегментов)
    """
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}")
        return
    
    # Получаем общее количество кадров
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Видео: {os.path.basename(video_path)}")
    print(f"  Всего кадров: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Длительность: {duration:.2f} секунд")
    
    # Вычисляем позиции кадров для равномерного распределения
    if total_frames <= num_frames:
        frame_positions = list(range(total_frames))
    else:
        # Разделить видео на num_frames равных сегментов
        segment_size = total_frames / num_frames
        frame_positions = []
        for i in range(num_frames):
            # Взять первый кадр из каждого сегмента
            frame_pos = int(i * segment_size)
            frame_positions.append(frame_pos)
    
    print(f"  Сегментов: {num_frames}, позиции кадров: {frame_positions[:3]}...{frame_positions[-2:] if len(frame_positions) > 5 else frame_positions}")
    
    # Создаем папку для этого видео
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    extracted_count = 0
    
    # Извлекаем кадры на вычисленных позициях
    for i, frame_pos in enumerate(frame_positions):
        # Переходим к нужному кадру
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        
        if not ret:
            print(f"    Предупреждение: не удалось прочитать кадр на позиции {frame_pos}")
            continue
            
        # Сохраняем кадр
        frame_filename = f"{video_name}_frame_{i + 1:02d}_pos_{frame_pos:06d}.jpg"
        frame_path = os.path.join(video_output_dir, frame_filename)
        
        success = cv2.imwrite(frame_path, frame)
        if success:
            extracted_count += 1
            if extracted_count <= 5 or extracted_count % 10 == 0:
                print(f"    Извлечен кадр {extracted_count}/{len(frame_positions)}: {frame_filename}")
        else:
            print(f"    Ошибка сохранения кадра: {frame_filename}")
    
    cap.release()
    print(f"  Завершено: извлечено {extracted_count} кадров\n")

def main():
    # Пути к папкам
    videos_dir = "Videos"
    output_dir = "Images"
    
    # Создаем папку для изображений если её нет
    os.makedirs(output_dir, exist_ok=True)
    
    # Ищем все видео файлы
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(videos_dir, ext)))
        video_files.extend(glob.glob(os.path.join(videos_dir, ext.upper())))
    
    if not video_files:
        print(f"Видео файлы не найдены в папке {videos_dir}")
        return
    
    print(f"Найдено {len(video_files)} видео файлов:")
    for video in video_files:
        print(f"  - {os.path.basename(video)}")
    print()
    
    # Обрабатываем каждое видео
    for video_path in video_files:
        try:
            extract_frames_from_video(video_path, output_dir, num_frames=30)
        except Exception as e:
            print(f"Ошибка при обработке {video_path}: {e}")
    
    print("Извлечение кадров завершено!")

if __name__ == "__main__":
    main() 