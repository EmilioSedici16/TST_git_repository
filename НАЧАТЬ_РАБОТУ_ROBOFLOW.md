# 🚀 НАЧАТЬ РАБОТУ: Полный Pipeline Roboflow

## 🎯 Что вы получили

**Объединенный проект** для полного pipeline компьютерного зрения:
1. **🎬 Извлечение кадров** из видео → подготовка данных
2. **📤 Загрузка в Roboflow** → разметка и обучение
3. **🤖 Детекция объектов** → применение моделей

## 🚀 БЫСТРЫЙ СТАРТ

### ▶️ Шаг 1: Объединить с Roboflow проектом
```
Дважды кликните: ОБЪЕДИНИТЬ_С_ROBOFLOW.bat
```

Выберите **Вариант 2** (объединить проекты)

### ▶️ Шаг 2: Запустить полный pipeline
```bash
# Автоматический pipeline
python ПОЛНЫЙ_PIPELINE.py --full

# Или по шагам:
python ПОЛНЫЙ_PIPELINE.py --extract-frames
python ПОЛНЫЙ_PIPELINE.py --prepare-roboflow
```

### ▶️ Шаг 3: Веб-интерфейс
```bash
python ПОЛНЫЙ_PIPELINE.py --web
```

## 📋 Пошаговый рабочий процесс

### 1️⃣ Подготовка данных
```bash
# Поместите видео в папку Videos/
cp your_videos/*.mp4 Videos/

# Извлеките кадры
python extract_frames.py
# ИЛИ
python ПОЛНЫЙ_PIPELINE.py --extract-frames
```

### 2️⃣ Работа с Roboflow
```bash
# Автоматическая подготовка
python ПОЛНЫЙ_PIPELINE.py --prepare-roboflow

# Если настроен API ключ - автозагрузка:
export ROBOFLOW_API_KEY="ваш_ключ"
python ПОЛНЫЙ_PIPELINE.py --upload-roboflow --project "your-project"
```

**Ручная работа в Roboflow:**
1. Перейдите на https://roboflow.com
2. Создайте проект
3. Загрузите кадры из папки `Images/`
4. Выполните разметку (аннотирование)
5. Настройте аугментацию
6. Обучите модель
7. Экспортируйте модель

### 3️⃣ Детекция
```bash
# С обученной моделью Roboflow
python detect.py --source Images/

# С YOLO моделью
python ПОЛНЫЙ_PIPELINE.py --detect --model "yolov8n.pt"

# Веб-интерфейс
streamlit run app.py
```

## 🗂️ Структура объединенного проекта

```
TST_Project/
├── 🎬 ИЗВЛЕЧЕНИЕ КАДРОВ:
│   ├── extract_frames.py          # Основной скрипт
│   ├── simple_extract.py          # Упрощенная версия
│   ├── Videos/                    # Входные видео
│   └── Images/                    # Извлеченные кадры
│
├── 🤖 ROBOFLOW & ДЕТЕКЦИЯ:
│   ├── app.py                     # Веб-интерфейс
│   ├── detect.py                  # Детекция объектов
│   ├── safety_detection.py       # Детекция безопасности
│   ├── utils.py                   # Утилиты
│   ├── yolov8n.pt                # Модель YOLO
│   ├── data/                      # Данные Roboflow
│   │   ├── raw/                   # Исходные данные
│   │   ├── processed/             # Обработанные
│   │   └── results/               # Результаты
│   └── models/                    # Обученные модели
│
├── 🚀 АВТОМАТИЗАЦИЯ:
│   ├── ПОЛНЫЙ_PIPELINE.py         # Основной pipeline
│   ├── ОБЪЕДИНИТЬ_С_ROBOFLOW.bat  # Скрипт объединения
│   └── run_extract.bat            # Быстрое извлечение
│
└── 📚 ДОКУМЕНТАЦИЯ:
    ├── README_ROBOFLOW_PIPELINE.md # Полное описание
    ├── ROBOFLOW_SETUP.md          # Настройка Roboflow
    └── QUICKSTART.md              # Быстрый старт
```

## 🎯 Примеры использования

### 📹 Обработка видео с камер безопасности
```bash
# 1. Извлечь кадры из записей
python extract_frames.py

# 2. Загрузить в Roboflow для разметки людей/объектов
python ПОЛНЫЙ_PIPELINE.py --upload-roboflow --project "security-detection"

# 3. После обучения - детекция на новых видео
python detect.py --source new_video.mp4
```

### 🏭 Контроль качества продукции
```bash
# 1. Извлечь кадры из видео производственной линии
python ПОЛНЫЙ_PIPELINE.py --extract-frames --num-frames 500

# 2. Разметить дефекты в Roboflow
# 3. Детекция дефектов
python safety_detection.py --source production_images/
```

### 🚗 Анализ дорожного движения
```bash
# 1. Кадры из видео с дорожных камер
python extract_frames.py

# 2. Разметка автомобилей, пешеходов в Roboflow
# 3. Анализ трафика
python detect.py --source traffic_video.mp4 --save-txt
```

## ⚙️ Настройки и конфигурация

### 🔧 Настройка извлечения кадров
```python
# В extract_frames.py или ПОЛНЫЙ_PIPELINE.py
NUM_FRAMES = 100          # Количество кадров
START_OFFSET = 0.1        # Пропуск начала (10%)
END_OFFSET = 0.1          # Пропуск конца (10%)
IMAGE_FORMAT = 'jpg'      # Формат изображений
```

### 🤖 Настройка детекции
```python
# В detect.py
CONFIDENCE_THRESHOLD = 0.5   # Порог уверенности
IOU_THRESHOLD = 0.45         # Порог IoU для NMS
MAX_DETECTIONS = 1000        # Максимум детекций
```

### 🌐 Настройка Roboflow
```bash
# API ключ
export ROBOFLOW_API_KEY="ваш_ключ"

# Или в коде
os.environ['ROBOFLOW_API_KEY'] = "ваш_ключ"
```

## 🔄 Синхронизация между компьютерами

### 📥 На втором ПК:
```bash
# Клонировать объединенный проект
git clone https://github.com/EmilioSedici16/TST_git_repository.git
cd TST_git_repository

# Установить зависимости
pip install -r requirements.txt

# Настроить Roboflow API
export ROBOFLOW_API_KEY="ваш_ключ"
```

### 🔄 Ежедневная работа:
```bash
# ПЕРЕД работой
git pull origin main

# ПОСЛЕ работы
git add .
git commit -m "Обновления pipeline"
git push origin main
```

### 📁 Что синхронизируется:
- ✅ Код и скрипты
- ✅ Конфигурация и настройки
- ✅ Документация
- ❌ Видеофайлы (используйте облако)
- ❌ Извлеченные кадры (генерируются локально)
- ❌ Модели (скачиваются с Roboflow)

## 🚨 Решение проблем

### ❌ Ошибки при объединении проектов
```bash
# Если возникли конфликты
git status
# Разрешите конфликты в файлах
git add .
git commit -m "Разрешение конфликтов"
git push origin main
```

### 🔑 Проблемы с Roboflow API
```bash
# Проверка ключа
python -c "import os; print(os.environ.get('ROBOFLOW_API_KEY'))"

# Тест подключения
python -c "from roboflow import Roboflow; rf = Roboflow(); print('OK')"
```

### 📦 Ошибки установки пакетов
```bash
# Обновление pip
python -m pip install --upgrade pip

# Переустановка зависимостей
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

## 📊 Мониторинг прогресса

### 📈 Логи pipeline
```bash
# Просмотр логов
tail -f pipeline.log

# Логи в реальном времени
python ПОЛНЫЙ_PIPELINE.py --full | tee pipeline.log
```

### 📊 Статистика проекта
```bash
# Количество извлеченных кадров
ls Images/*.jpg | wc -l

# Размер данных
du -sh Videos/ Images/ data/ models/

# Результаты детекции
ls data/results/
```

## 🎓 Следующие шаги

1. **🔍 Изучите** документацию в `README_ROBOFLOW_PIPELINE.md`
2. **🚀 Запустите** первый pipeline: `python ПОЛНЫЙ_PIPELINE.py --full`
3. **📤 Создайте** проект в Roboflow и выполните разметку
4. **🤖 Обучите** модель в Roboflow
5. **🎯 Протестируйте** детекцию на новых данных
6. **📈 Масштабируйте** на больше данных и классов

## 📞 Поддержка

- 📚 **Документация:** `README_ROBOFLOW_PIPELINE.md`
- 🐛 **Баги:** https://github.com/EmilioSedici16/TST_git_repository/issues
- 💡 **Идеи:** GitHub Discussions
- 🌐 **Roboflow:** https://docs.roboflow.com

---

**🎯 Начните с команды:** `python ПОЛНЫЙ_PIPELINE.py --full`

**🌐 Ваш GitHub:** https://github.com/EmilioSedici16/TST_git_repository