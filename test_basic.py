#!/usr/bin/env python3
"""
Простой тест основной функциональности
"""

import os
from utils import setup_roboflow_env, RoboflowManager, ImageProcessor
from ultralytics import YOLO

def test_environment():
    """Тестирует настройку окружения"""
    print("🔍 Тестирование окружения...")
    
    # Проверка API ключа
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if api_key:
        print("✅ API ключ Roboflow найден")
    else:
        print("❌ API ключ Roboflow НЕ найден")
        print("💡 Установите его командой: export ROBOFLOW_API_KEY='ваш_ключ'")
    
    # Проверка YOLOv8
    try:
        model = YOLO('yolov8n.pt')  # Загрузит модель автоматически
        print("✅ YOLOv8 работает корректно")
    except Exception as e:
        print(f"❌ Ошибка YOLOv8: {e}")
    
    # Проверка OpenCV
    try:
        import cv2
        print(f"✅ OpenCV версия: {cv2.__version__}")
    except Exception as e:
        print(f"❌ Ошибка OpenCV: {e}")

def test_yolo_detection():
    """Тестирует детекцию с помощью YOLOv8 на примере изображения"""
    print("\n🎯 Тестирование детекции объектов...")
    
    try:
        # Загружаем предобученную модель YOLOv8
        model = YOLO('yolov8n.pt')
        
        # Создаем тестовое изображение (просто цветной прямоугольник)
        import numpy as np
        import cv2
        
        # Создаем простое тестовое изображение
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[:] = (100, 150, 200)  # Заливаем цветом
        
        # Добавляем простые фигуры
        cv2.rectangle(test_img, (50, 50), (200, 200), (0, 255, 0), -1)
        cv2.circle(test_img, (400, 300), 80, (255, 0, 0), -1)
        
        # Сохраняем тестовое изображение
        cv2.imwrite('data/test_image.jpg', test_img)
        print("✅ Тестовое изображение создано: data/test_image.jpg")
        
        # Выполняем детекцию
        results = model('data/test_image.jpg')
        print(f"✅ Детекция выполнена успешно")
        print(f"📊 Найдено объектов: {len(results[0].boxes) if results[0].boxes is not None else 0}")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования детекции: {e}")

def main():
    """Основная функция"""
    print("🚀 Тестирование проекта компьютерного зрения с Roboflow\n")
    
    # Создаем папку для данных если её нет
    os.makedirs('data', exist_ok=True)
    
    # Тестируем окружение
    test_environment()
    
    # Тестируем детекцию
    test_yolo_detection()
    
    print("\n✨ Тестирование завершено!")
    print("\n📋 Следующие шаги:")
    print("1. Если все тесты прошли успешно - переходите к реальной детекции")
    print("2. Если есть ошибки - установите недостающие компоненты")
    print("3. Для работы с Roboflow получите API ключ на roboflow.com")

if __name__ == "__main__":
    main()