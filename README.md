# Проект компьютерного зрения с Roboflow

Этот проект демонстрирует использование Roboflow для задач компьютерного зрения, включая детекцию объектов и анализ изображений.

## Возможности

- 🤖 Интеграция с Roboflow API
- 🔍 Детекция объектов в реальном времени
- 📊 Визуализация результатов
- 🌐 Веб-интерфейс для демонстрации
- 📸 Обработка изображений и видео

## Установка

1. Клонируйте репозиторий
2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Настройте API ключ Roboflow:
```bash
export ROBOFLOW_API_KEY="ваш_api_ключ"
```

## Использование

### Базовая детекция
```bash
python detect.py --source path/to/image.jpg
```

### Веб-приложение
```bash
streamlit run app.py
```

## Структура проекта

```
├── models/          # Модели и веса
├── data/           # Датасеты и изображения
├── scripts/        # Вспомогательные скрипты
├── detect.py       # Основной скрипт детекции
├── app.py          # Веб-приложение
└── utils.py        # Утилиты
```

## Требования

- Python 3.8+
- Аккаунт в Roboflow
- API ключ Roboflow