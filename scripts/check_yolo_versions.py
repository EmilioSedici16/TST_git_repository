#!/usr/bin/env python3
"""
Скрипт для проверки доступных версий YOLO и актуализации стека
"""

import sys
import subprocess
from pathlib import Path

def check_ultralytics_version():
    """Проверка текущей версии ultralytics"""
    try:
        import ultralytics
        return ultralytics.__version__
    except ImportError:
        return None

def get_available_yolo_models():
    """Список доступных моделей YOLO"""
    models = {
        'YOLOv8': {
            'description': 'Стабильная версия (выпущена 10 января 2023)',
            'models': {
                'yolov8n.pt': 'Nano - самая быстрая, наименьший размер',
                'yolov8s.pt': 'Small - баланс скорости и точности',
                'yolov8m.pt': 'Medium - хорошая точность',
                'yolov8l.pt': 'Large - высокая точность',
                'yolov8x.pt': 'XLarge - максимальная точность'
            },
            'specialized': {
                'yolov8n-seg.pt': 'Сегментация экземпляров (nano)',
                'yolov8s-seg.pt': 'Сегментация экземпляров (small)',
                'yolov8m-seg.pt': 'Сегментация экземпляров (medium)',
                'yolov8l-seg.pt': 'Сегментация экземпляров (large)',
                'yolov8x-seg.pt': 'Сегментация экземпляров (xlarge)',
                'yolov8n-pose.pt': 'Оценка позы/ключевые точки (nano)',
                'yolov8s-pose.pt': 'Оценка позы/ключевые точки (small)',
                'yolov8m-pose.pt': 'Оценка позы/ключевые точки (medium)',
                'yolov8l-pose.pt': 'Оценка позы/ключевые точки (large)',
                'yolov8x-pose.pt': 'Оценка позы/ключевые точки (xlarge)',
                'yolov8n-cls.pt': 'Классификация изображений (nano)',
                'yolov8s-cls.pt': 'Классификация изображений (small)',
                'yolov8m-cls.pt': 'Классификация изображений (medium)',
                'yolov8l-cls.pt': 'Классификация изображений (large)',
                'yolov8x-cls.pt': 'Классификация изображений (xlarge)'
            }
        },
        'YOLOv9': {
            'description': 'Новая архитектура с улучшенной точностью',
            'models': {
                'yolov9t.pt': 'Tiny - компактная версия',
                'yolov9s.pt': 'Small',
                'yolov9m.pt': 'Medium',
                'yolov9c.pt': 'C - улучшенная версия',
                'yolov9e.pt': 'E - расширенная версия'
            }
        },
        'YOLOv10': {
            'description': 'Оптимизированная версия с улучшенной производительностью',
            'models': {
                'yolov10n.pt': 'Nano',
                'yolov10s.pt': 'Small',
                'yolov10m.pt': 'Medium',
                'yolov10b.pt': 'Base',
                'yolov10l.pt': 'Large',
                'yolov10x.pt': 'XLarge'
            }
        },
        'YOLOv11': {
            'description': 'Последняя стабильная версия для production',
            'models': {
                'yolov11n.pt': 'Nano',
                'yolov11s.pt': 'Small',
                'yolov11m.pt': 'Medium',
                'yolov11l.pt': 'Large',
                'yolov11x.pt': 'XLarge'
            }
        }
    }
    return models

def check_pip_latest_version():
    """Проверка последней версии ultralytics на PyPI"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'index', 'versions', 'ultralytics'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Парсим вывод для получения последней версии
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Available versions:' in line or 'versions:' in line.lower():
                    # Извлекаем версии
                    return line
        return None
    except Exception as e:
        return f"Ошибка: {e}"

def print_report():
    """Вывод отчета о доступных версиях YOLO"""
    print("=" * 70)
    print("ПРОВЕРКА ДОСТУПНЫХ ВЕРСИЙ YOLO ДЛЯ АКТУАЛИЗАЦИИ СТЕКА")
    print("=" * 70)
    
    # Текущая версия
    current_version = check_ultralytics_version()
    if current_version:
        print(f"\n✅ Текущая версия ultralytics: {current_version}")
    else:
        print("\n❌ Библиотека ultralytics не установлена")
    
    # Доступные модели
    models = get_available_yolo_models()
    
    print("\n" + "=" * 70)
    print("ДОСТУПНЫЕ ВЕРСИИ YOLO И МОДЕЛИ")
    print("=" * 70)
    
    for version_name, version_info in models.items():
        print(f"\n📦 {version_name}")
        print(f"   Описание: {version_info['description']}")
        print("\n   Основные модели детекции:")
        for model_name, model_desc in version_info['models'].items():
            print(f"      • {model_name:20s} - {model_desc}")
        
        if 'specialized' in version_info:
            print("\n   Специализированные модели:")
            for model_name, model_desc in version_info['specialized'].items():
                print(f"      • {model_name:20s} - {model_desc}")
    
    print("\n" + "=" * 70)
    print("РЕКОМЕНДАЦИИ ПО ОБНОВЛЕНИЮ")
    print("=" * 70)
    
    print("\n1. YOLOv8 (рекомендуется для вашего проекта):")
    print("   ✅ Стабильная и проверенная версия")
    print("   ✅ Хорошая поддержка в ultralytics")
    print("   ✅ Подходит для детекции людей и касок")
    print("   ✅ Модели: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt")
    
    print("\n2. YOLOv9 (для максимальной точности):")
    print("   ⚠️  Новая архитектура, может потребовать обновления кода")
    print("   ✅ Улучшенная точность детекции")
    print("   ⚠️  Меньше документации и примеров")
    
    print("\n3. YOLOv10 (для оптимизации производительности):")
    print("   ✅ Оптимизированная производительность")
    print("   ⚠️  Относительно новая версия")
    
    print("\n4. YOLOv11 (для production):")
    print("   ✅ Рекомендуется для стабильных production систем")
    print("   ✅ Последняя стабильная версия")
    
    print("\n" + "=" * 70)
    print("ТЕКУЩЕЕ ИСПОЛЬЗОВАНИЕ В ПРОЕКТЕ")
    print("=" * 70)
    
    # Проверяем, какие модели используются в проекте
    project_files = ['detect.py', 'safety_detection.py', 'app.py']
    used_models = set()
    
    for file_path in project_files:
        file = Path(file_path)
        if file.exists():
            content = file.read_text(encoding='utf-8')
            # Ищем упоминания моделей
            for model in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']:
                if model in content:
                    used_models.add(f"{model}.pt")
    
    if used_models:
        print("\n📋 Используемые модели в проекте:")
        for model in sorted(used_models):
            print(f"   • {model}")
    else:
        print("\n⚠️  Не удалось определить используемые модели")
    
    print("\n" + "=" * 70)
    print("КОМАНДЫ ДЛЯ ОБНОВЛЕНИЯ")
    print("=" * 70)
    
    print("\n1. Обновить библиотеку ultralytics:")
    print("   pip install --upgrade ultralytics")
    
    print("\n2. Скачать новую модель YOLOv8:")
    print("   python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
    
    print("\n3. Скачать модель YOLOv9:")
    print("   python -c \"from ultralytics import YOLO; YOLO('yolov9t.pt')\"")
    
    print("\n4. Скачать модель YOLOv10:")
    print("   python -c \"from ultralytics import YOLO; YOLO('yolov10n.pt')\"")
    
    print("\n5. Скачать модель YOLOv11:")
    print("   python -c \"from ultralytics import YOLO; YOLO('yolov11n.pt')\"")
    
    print("\n" + "=" * 70)
    print("РЕКОМЕНДАЦИЯ ДЛЯ ПРОЕКТА ДЕТЕКЦИИ КАСОК")
    print("=" * 70)
    
    print("\n💡 Для вашего проекта рекомендуется:")
    print("   1. Остаться на YOLOv8 (стабильная версия)")
    print("   2. Использовать yolov8m.pt или yolov8l.pt для лучшей точности")
    print("   3. Обновить ultralytics до последней версии 8.x")
    print("   4. Рассмотреть YOLOv9 или YOLOv11 для максимальной точности")
    print("      (после тестирования совместимости)")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        print_report()
    except Exception as e:
        print(f"❌ Ошибка при выполнении проверки: {e}")
        import traceback
        traceback.print_exc()
