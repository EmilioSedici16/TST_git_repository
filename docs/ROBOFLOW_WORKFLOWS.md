# 🔄 Работа с Roboflow Workflows

## Обзор

Roboflow Workflows позволяют запускать сложные пайплайны детекции через API. Для проекта детекции касок используется workflow `find-people-helmets-and-lifts`.

## Проблема с inference-sdk

Библиотека `inference-sdk` **несовместима с Python 3.13** (требует Python <3.13).

## Решение: RoboflowWorkflowClient

Создан альтернативный HTTP клиент `RoboflowWorkflowClient`, который:
- ✅ Работает с Python 3.13+
- ✅ Не требует установки `inference-sdk`
- ✅ Использует прямые HTTP запросы к Roboflow API
- ✅ Полностью совместим с API Roboflow Workflows

## Установка

Никаких дополнительных установок не требуется! `RoboflowWorkflowClient` использует только стандартные библиотеки:
- `requests` (уже в requirements.txt)
- `base64` (встроенная библиотека)

## Использование

### Базовый пример

```python
from roboflow_workflow_client import RoboflowWorkflowClient

# Инициализация клиента
client = RoboflowWorkflowClient(
    api_url="https://serverless.roboflow.com",
    api_key="WJqw53R6TlCZVbqB3fdU"  # Ваш API ключ
)

# Запуск workflow
result = client.run_workflow(
    workspace_name="tst-workspace",
    workflow_id="find-people-helmets-and-lifts",
    images={
        "image": "path/to/image.jpg"
    },
    use_cache=True  # Кэшировать определение workflow на 15 минут
)

print(result)
```

### Использование переменной окружения для API ключа

```python
import os

# Установите переменную окружения
os.environ['ROBOFLOW_API_KEY'] = "WJqw53R6TlCZVbqB3fdU"

# Клиент автоматически использует переменную окружения
client = RoboflowWorkflowClient()
```

### Работа с изображениями по URL

```python
result = client.run_workflow_from_url(
    workspace_name="tst-workspace",
    workflow_id="find-people-helmets-and-lifts",
    image_urls={
        "image": "https://example.com/image.jpg"
    },
    use_cache=True
)
```

## Интеграция с проектом

### Пример использования в safety_detection.py

```python
from roboflow_workflow_client import RoboflowWorkflowClient

class SafetyDetector:
    def __init__(self):
        self.workflow_client = RoboflowWorkflowClient(
            api_key=os.getenv('ROBOFLOW_API_KEY')
        )
    
    def detect_with_workflow(self, image_path):
        """Детекция через Roboflow Workflow"""
        result = self.workflow_client.run_workflow(
            workspace_name="tst-workspace",
            workflow_id="find-people-helmets-and-lifts",
            images={"image": image_path},
            use_cache=True
        )
        return result
```

## Структура ответа Workflow

Workflow возвращает JSON с результатами детекции:

```json
{
  "predictions": [
    {
      "class": "person",
      "confidence": 0.95,
      "x": 100,
      "y": 200,
      "width": 50,
      "height": 100
    },
    {
      "class": "helmet",
      "confidence": 0.87,
      "x": 110,
      "y": 190,
      "width": 30,
      "height": 30
    }
  ],
  "workflow_id": "find-people-helmets-and-lifts",
  "processing_time": 1.23
}
```

## Преимущества Workflows

1. **Готовые пайплайны** - не нужно настраивать модели отдельно
2. **Комбинированная детекция** - находит людей, каски и подъемные платформы одновременно
3. **Обновления на сервере** - улучшения применяются автоматически
4. **Кэширование** - определение workflow кэшируется на 15 минут

## Ограничения

1. **Требует интернет** - все запросы идут через API
2. **Лимиты API** - могут быть ограничения на количество запросов
3. **Задержка сети** - обработка занимает больше времени из-за сетевых запросов

## Сравнение с локальными моделями

| Характеристика | Workflows | Локальные модели |
|----------------|-----------|------------------|
| Скорость | ⚠️ Зависит от сети | ✅ Быстро |
| Точность | ✅ Высокая | ✅ Высокая |
| Офлайн работа | ❌ Нет | ✅ Да |
| Обновления | ✅ Автоматически | ⚠️ Вручную |
| Настройка | ✅ Не требуется | ⚠️ Требуется |

## Примеры

Полные примеры использования находятся в:
- `examples/roboflow_workflow_example.py` - базовые примеры
- `roboflow_workflow_client.py` - исходный код клиента

## Документация API

Официальная документация Roboflow Workflows:
- https://docs.roboflow.com/inference/workflows

---

*Документация создана: 2024*
*Альтернатива inference-sdk для Python 3.13+*
