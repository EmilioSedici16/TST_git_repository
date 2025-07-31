#!/usr/bin/env python3
"""
Основной скрипт детекции объектов с использованием YOLOv8 и Roboflow
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from utils import ImageProcessor, Visualizer, RoboflowManager
import time

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Детекция объектов с YOLOv8 и Roboflow')
    
    parser.add_argument('--source', type=str, default='0',
                       help='Источник: путь к изображению, видео или 0 для веб-камеры')
    
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Путь к модели или название предобученной модели')
    
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Порог уверенности (0.0-1.0)')
    
    parser.add_argument('--save', action='store_true',
                       help='Сохранить результаты')
    
    parser.add_argument('--output', type=str, default='runs/detect',
                       help='Папка для сохранения результатов')
    
    parser.add_argument('--show', action='store_true',
                       help='Показать результаты на экране')
    
    parser.add_argument('--roboflow-workspace', type=str,
                       help='Workspace в Roboflow для использования кастомной модели')
    
    parser.add_argument('--roboflow-project', type=str,
                       help='Проект в Roboflow')
    
    parser.add_argument('--roboflow-version', type=int, default=1,
                       help='Версия модели в Roboflow')
    
    return parser.parse_args()

def load_model(args):
    """Загрузка модели"""
    print(f"🔄 Загрузка модели...")
    
    # Если указаны параметры Roboflow, пытаемся использовать модель оттуда
    if args.roboflow_workspace and args.roboflow_project:
        try:
            rf_manager = RoboflowManager()
            if rf_manager.api_key:
                model = rf_manager.get_model(
                    args.roboflow_workspace, 
                    args.roboflow_project, 
                    args.roboflow_version
                )
                if model:
                    print(f"✅ Загружена модель Roboflow: {args.roboflow_project}")
                    return model
        except Exception as e:
            print(f"⚠️ Не удалось загрузить модель Roboflow: {e}")
            print("🔄 Переключаемся на локальную модель YOLOv8...")
    
    # Загружаем локальную модель YOLOv8
    try:
        model = YOLO(args.model)
        print(f"✅ Загружена модель: {args.model}")
        return model
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None

def detect_image(model, image_path, args):
    """Детекция объектов на изображении"""
    print(f"🖼️ Обработка изображения: {image_path}")
    
    # Загружаем изображение
    image = ImageProcessor.load_image(image_path)
    if image is None:
        print(f"❌ Не удалось загрузить изображение: {image_path}")
        return None
    
    # Выполняем детекцию
    results = model(image, conf=args.conf)
    
    # Обрабатываем результаты
    detections = []
    for r in results:
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()  # координаты bbox
            confs = r.boxes.conf.cpu().numpy()  # уверенность
            classes = r.boxes.cls.cpu().numpy().astype(int)  # классы
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                detections.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class': int(cls)
                })
    
    print(f"📊 Найдено объектов: {len(detections)}")
    
    # Визуализация
    if detections:
        # Получаем названия классов
        class_names = model.names if hasattr(model, 'names') else None
        
        # Рисуем рамки
        result_image = ImageProcessor.draw_bboxes(image, detections, class_names)
        
        # Показываем результат
        if args.show:
            cv2.imshow('Детекция объектов', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Сохраняем результат
        if args.save:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"result_{Path(image_path).name}"
            ImageProcessor.save_image(result_image, str(output_path))
            print(f"💾 Результат сохранен: {output_path}")
        
        return result_image, detections
    
    return image, []

def detect_video(model, video_path, args):
    """Детекция объектов в видео или с веб-камеры"""
    # Определяем источник видео
    if video_path == '0':
        cap = cv2.VideoCapture(0)  # Веб-камера
        print("📹 Запуск детекции с веб-камеры...")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"🎬 Обработка видео: {video_path}")
    
    if not cap.isOpened():
        print("❌ Не удалось открыть источник видео")
        return
    
    # Получаем параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Настройка записи видео (если нужно сохранить)
    out = None
    if args.save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"result_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"💾 Видео будет сохранено: {output_path}")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Выполняем детекцию
            results = model(frame, conf=args.conf, verbose=False)
            
            # Обрабатываем результаты
            detections = []
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, cls in zip(boxes, confs, classes):
                        detections.append({
                            'bbox': box.tolist(),
                            'confidence': float(conf),
                            'class': int(cls)
                        })
            
            # Рисуем результаты
            class_names = model.names if hasattr(model, 'names') else None
            result_frame = ImageProcessor.draw_bboxes(frame, detections, class_names)
            
            # Добавляем информацию на кадр
            cv2.putText(result_frame, f"Objects: {len(detections)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Frame: {frame_count}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Показываем кадр
            if args.show:
                cv2.imshow('Детекция объектов (нажмите Q для выхода)', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Сохраняем кадр в видео
            if out is not None:
                out.write(result_frame)
            
            # Выводим прогресс каждые 30 кадров
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_real = frame_count / elapsed
                print(f"🎬 Обработано кадров: {frame_count}, FPS: {fps_real:.1f}")
    
    except KeyboardInterrupt:
        print("⏹️ Остановлено пользователем")
    
    finally:
        # Освобождаем ресурсы
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        if frame_count > 0:
            print(f"✅ Обработка завершена: {frame_count} кадров за {elapsed:.1f}с")

def main():
    """Основная функция"""
    args = parse_arguments()
    
    print("🚀 Система детекции объектов")
    print(f"📂 Источник: {args.source}")
    print(f"🎯 Порог уверенности: {args.conf}")
    print("-" * 50)
    
    # Загружаем модель
    model = load_model(args)
    if model is None:
        print("❌ Не удалось загрузить модель")
        return
    
    # Определяем тип источника и запускаем соответствующую обработку
    source = args.source
    
    # Проверяем, является ли источник изображением
    if source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
        detect_image(model, source, args)
    
    # Проверяем, является ли источник URL изображения
    elif source.startswith(('http://', 'https://')) and any(source.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
        detect_image(model, source, args)
    
    # Иначе обрабатываем как видео или веб-камеру
    else:
        detect_video(model, source, args)
    
    print("✨ Работа завершена!")

if __name__ == "__main__":
    main()