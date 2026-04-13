#!/usr/bin/env python3
"""
Пример использования supervision для визуализации детекций
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from utils import ImageProcessor, convert_ultralytics_to_supervision, SUPERVISION_AVAILABLE

if not SUPERVISION_AVAILABLE:
    print("❌ Supervision не установлен. Установите: pip install supervision")
    exit(1)

import supervision as sv

def example_supervision_visualization():
    """Пример визуализации с использованием supervision"""
    
    # Загружаем модель
    model = YOLO('yolov8n.pt')
    
    # Загружаем изображение
    image_path = 'data/test_image.jpg'
    if not Path(image_path).exists():
        print(f"⚠️ Изображение не найдено: {image_path}")
        print("💡 Используйте любое изображение для теста")
        return
    
    image = ImageProcessor.load_image(image_path)
    if image is None:
        print(f"❌ Не удалось загрузить изображение: {image_path}")
        return
    
    # Выполняем детекцию
    results = model(image, conf=0.5)
    
    # Конвертируем в формат supervision
    detections = convert_ultralytics_to_supervision(results, model.names)
    
    if detections is None or len(detections) == 0:
        print("⚠️ Детекции не найдены")
        return
    
    # Создаем аннотаторы
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Формируем метки
    labels = []
    for i, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
        class_name = model.names.get(int(class_id), f"Class {class_id}")
        labels.append(f"{class_name} {confidence:.2f}")
    
    # Аннотируем изображение
    annotated_image = box_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels
    )
    
    # Показываем результат
    cv2.imshow('Supervision Visualization', annotated_image)
    print("✅ Нажмите любую клавишу для закрытия окна")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Сохраняем результат
    output_path = 'runs/supervision_example.jpg'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, annotated_image)
    print(f"💾 Результат сохранен: {output_path}")

def example_filter_by_class():
    """Пример фильтрации детекций по классу (люди)"""
    
    model = YOLO('yolov8n.pt')
    image_path = 'data/test_image.jpg'
    
    if not Path(image_path).exists():
        print(f"⚠️ Изображение не найдено: {image_path}")
        return
    
    image = ImageProcessor.load_image(image_path)
    if image is None:
        return
    
    results = model(image, conf=0.5)
    detections = convert_ultralytics_to_supervision(results, model.names)
    
    if detections is None:
        return
    
    # Фильтруем только людей (class_id = 0 в COCO)
    people_detections = detections[detections.class_id == 0]
    
    print(f"📊 Всего детекций: {len(detections)}")
    print(f"👥 Людей найдено: {len(people_detections)}")
    
    # Визуализируем только людей
    box_annotator = sv.BoxAnnotator()
    annotated_image = box_annotator.annotate(
        scene=image.copy(),
        detections=people_detections
    )
    
    cv2.imshow('People Only', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🎨 Пример использования Supervision для визуализации")
    print("-" * 60)
    
    try:
        example_supervision_visualization()
        print("\n" + "-" * 60)
        example_filter_by_class()
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
