#!/usr/bin/env python3
"""
Пример использования Roboflow Workflows для детекции людей, касок и подъемных платформ
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from roboflow_workflow_client import RoboflowWorkflowClient
from utils import ImageProcessor
import json

def example_workflow_detection():
    """Пример детекции через Roboflow Workflow"""
    
    # Инициализация клиента
    # API ключ можно установить через переменную окружения ROBOFLOW_API_KEY
    # или передать напрямую
    client = RoboflowWorkflowClient(
        api_url="https://serverless.roboflow.com",
        api_key="WJqw53R6TlCZVbqB3fdU"  # Замените на ваш ключ
    )
    
    # Путь к изображению
    image_path = "data/test_image.jpg"
    
    if not Path(image_path).exists():
        print(f"⚠️ Изображение не найдено: {image_path}")
        print("💡 Используйте любое изображение для теста")
        return
    
    print(f"🔄 Запуск workflow для изображения: {image_path}")
    print("-" * 60)
    
    try:
        # Запускаем workflow
        result = client.run_workflow(
            workspace_name="tst-workspace",
            workflow_id="find-people-helmets-and-lifts",
            images={
                "image": image_path
            },
            use_cache=True  # Кэшировать определение workflow на 15 минут
        )
        
        print("✅ Workflow выполнен успешно!")
        print("\n📊 Результаты:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Сохраняем результат
        output_path = "runs/workflow_result.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Результат сохранен: {output_path}")
        
    except Exception as e:
        print(f"❌ Ошибка выполнения workflow: {e}")
        import traceback
        traceback.print_exc()

def example_workflow_from_url():
    """Пример использования workflow с изображением по URL"""
    
    client = RoboflowWorkflowClient(
        api_url="https://serverless.roboflow.com",
        api_key="WJqw53R6TlCZVbqB3fdU"
    )
    
    # Пример URL изображения
    image_url = "https://ultralytics.com/images/bus.jpg"
    
    print(f"🔄 Запуск workflow для URL: {image_url}")
    print("-" * 60)
    
    try:
        result = client.run_workflow_from_url(
            workspace_name="tst-workspace",
            workflow_id="find-people-helmets-and-lifts",
            image_urls={
                "image": image_url
            },
            use_cache=True
        )
        
        print("✅ Workflow выполнен успешно!")
        print("\n📊 Результаты:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    print("🚀 Пример использования Roboflow Workflows")
    print("=" * 60)
    
    # Пример 1: Локальное изображение
    print("\n[Пример 1] Детекция на локальном изображении")
    example_workflow_detection()
    
    # Пример 2: Изображение по URL (раскомментируйте для использования)
    # print("\n[Пример 2] Детекция на изображении по URL")
    # example_workflow_from_url()
