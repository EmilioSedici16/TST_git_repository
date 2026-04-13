#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для анализа применимости Docker CV-окружения к проекту
"""

import sys
import io

# Настройка кодировки для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def analyze_components():
    """Анализ компонентов Docker-окружения"""
    
    components = {
        "YOLOv8/11 (Ultralytics)": {
            "status": "✅ Уже используется",
            "priority": "⭐⭐⭐⭐⭐",
            "applicability": "Полностью применимо",
            "recommendation": "Продолжать использовать, Docker ускорит на GPU",
            "action": "Никаких действий не требуется"
        },
        "Supervision": {
            "status": "🔥 Очень полезно",
            "priority": "⭐⭐⭐⭐⭐",
            "applicability": "Высокая - упростит визуализацию и обработку",
            "recommendation": "✅ Добавить немедленно",
            "action": "pip install supervision"
        },
        "RF-DETR (Roboflow)": {
            "status": "🔥 Полезно для точности",
            "priority": "⭐⭐⭐⭐",
            "applicability": "Высокая - улучшит детекцию касок",
            "recommendation": "✅ Добавить для экспериментов",
            "action": "Интегрировать через RoboflowManager"
        },
        "Grounding DINO 1.0": {
            "status": "💡 Полезно опционально",
            "priority": "⭐⭐⭐",
            "applicability": "Средняя - для валидации результатов",
            "recommendation": "⚠️ Опционально",
            "action": "Только для анализа статических изображений"
        },
        "SAM 2 (Meta)": {
            "status": "💡 Для продвинутых задач",
            "priority": "⭐⭐⭐",
            "applicability": "Средняя - точная сегментация",
            "recommendation": "⚠️ Для продвинутых задач",
            "action": "Если нужна точная сегментация границ"
        },
        "Grounding DINO 1.5 (API)": {
            "status": "⚠️ Ограниченная применимость",
            "priority": "⭐⭐",
            "applicability": "Низкая - требует интернет",
            "recommendation": "❌ Не рекомендуется",
            "action": "Лучше использовать локальную версию 1.0"
        },
        "OpenCV": {
            "status": "✅ Уже используется",
            "priority": "⭐⭐⭐⭐⭐",
            "applicability": "Полностью применимо",
            "recommendation": "Продолжать использовать",
            "action": "Docker даст GPU ускорение"
        },
        "Inference (Roboflow)": {
            "status": "✅ Частично используется",
            "priority": "⭐⭐⭐⭐",
            "applicability": "Высокая - расширить использование",
            "recommendation": "✅ Расширить использование",
            "action": "pip install inference inference-gpu"
        },
        "Jupyter": {
            "status": "💡 Полезно для разработки",
            "priority": "⭐⭐⭐",
            "applicability": "Средняя - для экспериментов",
            "recommendation": "✅ Для разработки",
            "action": "pip install jupyter"
        }
    }
    
    return components

def print_analysis():
    """Вывод анализа"""
    print("=" * 80)
    print("АНАЛИЗ ПРИМЕНИМОСТИ DOCKER CV-ОКРУЖЕНИЯ ДЛЯ ПРОЕКТА ДЕТЕКЦИИ КАСОК")
    print("=" * 80)
    
    components = analyze_components()
    
    print("\n📋 КОМПОНЕНТЫ DOCKER-ОКРУЖЕНИЯ:")
    print("-" * 80)
    
    for name, info in components.items():
        print(f"\n🔹 {name}")
        print(f"   Статус: {info['status']}")
        print(f"   Приоритет: {info['priority']}")
        print(f"   Применимость: {info['applicability']}")
        print(f"   Рекомендация: {info['recommendation']}")
        print(f"   Действие: {info['action']}")
    
    print("\n" + "=" * 80)
    print("ТОП-3 РЕКОМЕНДАЦИИ")
    print("=" * 80)
    
    print("\n1. 🔥 Supervision (Критически важно)")
    print("   ✅ Упростит визуализацию детекций")
    print("   ✅ Улучшит обработку видео")
    print("   ✅ Заменит ручную отрисовку в ImageProcessor")
    print("   Команда: pip install supervision")
    
    print("\n2. 🔥 RF-DETR (Высокий приоритет)")
    print("   ✅ Улучшит точность детекции касок")
    print("   ✅ Легко интегрируется через Roboflow")
    print("   ✅ Можно использовать как дополнительную модель")
    
    print("\n3. 💡 Расширение Inference (Roboflow)")
    print("   ✅ Полная библиотека inference")
    print("   ✅ Улучшенная работа с GPU")
    print("   Команда: pip install inference inference-gpu")
    
    print("\n" + "=" * 80)
    print("DOCKER-ОКРУЖЕНИЕ")
    print("=" * 80)
    
    print("\n✅ Преимущества:")
    print("   • Изоляция окружения")
    print("   • GPU ускорение (CUDA 12.1)")
    print("   • Простота развертывания")
    print("   • Оптимизированный PyTorch 2.2.0")
    
    print("\n⚠️ Когда использовать:")
    print("   • Production развертывание")
    print("   • Серверная обработка на GPU")
    print("   • Командная работа")
    
    print("\n❌ Когда НЕ использовать:")
    print("   • Быстрое тестирование")
    print("   • Разработка на Windows без GPU")
    print("   • Простая детекция (текущее окружение достаточно)")
    
    print("\n" + "=" * 80)
    print("ПЛАН ВНЕДРЕНИЯ")
    print("=" * 80)
    
    print("\nЭтап 1 (Немедленно):")
    print("   1. pip install supervision")
    print("   2. Обновить ImageProcessor для использования supervision")
    
    print("\nЭтап 2 (В ближайшее время):")
    print("   1. Интегрировать RF-DETR через RoboflowManager")
    print("   2. pip install inference inference-gpu")
    print("   3. Создать тестовый скрипт для сравнения моделей")
    
    print("\nЭтап 3 (Опционально):")
    print("   1. Grounding DINO 1.0 для валидации")
    print("   2. SAM 2 для точной сегментации")
    print("   3. Jupyter для экспериментов")
    
    print("\n" + "=" * 80)
    print("Подробный отчет: docs/DOCKER_CV_ANALYSIS.md")
    print("=" * 80)

if __name__ == "__main__":
    try:
        print_analysis()
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
