# 🎬🤖 TST Project: Полный Pipeline Компьютерного Зрения

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![Roboflow](https://img.shields.io/badge/Roboflow-Integration-orange.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-red.svg)

## 🎯 Описание проекта

Полный pipeline для компьютерного зрения: от извлечения кадров из видео до детекции объектов с помощью Roboflow и YOLO.

### 📋 Рабочий процесс:
1. **🎬 Извлечение кадров** из видеофайлов
2. **📤 Загрузка в Roboflow** для разметки и обучения
3. **🤖 Детекция объектов** с помощью обученных моделей
4. **📊 Анализ результатов** и визуализация

## 🏗️ Структура проекта

```
TST_Project/
├── 🎬 ЭТАП 1: Извлечение кадров
│   ├── extract_frames.py          # Основной скрипт извлечения
│   ├── simple_extract.py          # Упрощенная версия
│   ├── Videos/                    # 📹 Входные видеофайлы
│   │   ├── *.mp4                  # Видео для обработки
│   │   └── .gitkeep              # Сохранение структуры
│   ├── Images/                    # 🖼️ Извлеченные кадры
│   │   └── .gitkeep              # Сохранение структуры
│   ├── run_extract.bat           # Быстрый запуск (Windows)
│   └── run_extract.ps1           # PowerShell версия
│
├── 🤖 ЭТАП 2: Roboflow интеграция
│   ├── app.py                     # 🌐 Веб-интерфейс Streamlit
│   ├── safety_detection.py       # 🔍 Детекция безопасности
│   ├── detect.py                  # 🎯 Основная детекция
│   ├── utils.py                   # 🛠️ Утилиты
│   ├── data/                      # 📊 Датасеты и результаты
│   └── yolov8n.pt                # 🧠 Предобученная модель YOLO
│
├── 📚 Документация
│   ├── README.md                  # Основная документация
│   ├── ROBOFLOW_SETUP.md         # Настройка Roboflow
│   ├── QUICKSTART.md             # Быстрый старт
│   └── ИНСТРУКЦИЯ_ЗАПУСКА.md     # Инструкции на русском
│
└── ⚙️ Конфигурация
    ├── requirements.txt           # Python зависимости
    ├── .gitignore                # Git исключения
    └── cursorrules               # Настройки Cursor
```

## 🚀 Быстрый старт

### 1️⃣ Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2️⃣ Настройка Roboflow API
```bash
# Получите API ключ на roboflow.com
export ROBOFLOW_API_KEY="ваш_api_ключ"
```

### 3️⃣ Полный pipeline

#### Шаг 1: Извлечение кадров
```bash
# Поместите видео в папку Videos/
python extract_frames.py

# Или упрощенная версия
python simple_extract.py

# Или через батник
run_extract.bat
```

#### Шаг 2: Работа с Roboflow
```bash
# Запуск веб-интерфейса
streamlit run app.py

# Или прямая детекция
python detect.py --source path/to/image.jpg
```

## 🎬 Этап 1: Извлечение кадров

### 🎯 Возможности
- Извлечение 100 кадров из каждого видео
- Автоматическая обработка всех видео в папке
- Сохранение с осмысленными именами файлов
- Пропуск уже обработанных видео

### 🔧 Использование
```bash
# Основной скрипт с полным функционалом
python extract_frames.py

# Простая версия для быстрого извлечения
python simple_extract.py

# Автоматический запуск с установкой зависимостей
auto_install_and_run.bat
```

### ⚙️ Настройки
```python
# В extract_frames.py можно изменить:
NUM_FRAMES = 100        # Количество кадров
INPUT_FOLDER = "Videos" # Папка с видео
OUTPUT_FOLDER = "Images" # Папка для кадров
```

## 🤖 Этап 2: Roboflow и детекция

### 🌐 Веб-интерфейс (Streamlit)
```bash
streamlit run app.py
```

Возможности:
- Загрузка изображений или видео
- Выбор моделей детекции
- Настройка параметров
- Визуализация результатов
- Экспорт результатов

### 🎯 Прямая детекция
```bash
# Детекция на изображении
python detect.py --source image.jpg

# Детекция на видео
python detect.py --source video.mp4

# Детекция с веб-камеры
python detect.py --source 0

# Пакетная обработка папки
python detect.py --source Images/
```

### 🔧 Настройка моделей
```python
# В detect.py или app.py
CONFIDENCE_THRESHOLD = 0.5  # Порог уверенности
MODEL_PATH = "yolov8n.pt"   # Путь к модели
ROBOFLOW_MODEL = "your-model-id"  # ID модели Roboflow
```

## 📊 Интеграция с Roboflow

### 🔑 Настройка API
1. Зарегистрируйтесь на [roboflow.com](https://roboflow.com)
2. Создайте проект
3. Получите API ключ
4. Настройте в переменных окружения:
   ```bash
   export ROBOFLOW_API_KEY="ваш_ключ"
   ```

### 📤 Загрузка данных
```python
from roboflow import Roboflow

rf = Roboflow(api_key="ваш_ключ")
project = rf.workspace().project("имя_проекта")

# Загрузка извлеченных кадров
project.upload("Images/", batch_name="extracted_frames")
```

### 🧠 Использование обученной модели
```python
# Загрузка модели из Roboflow
model = project.version(1).model

# Предсказание
prediction = model.predict("path/to/image.jpg")
print(prediction.json())
```

## 🔄 Полный рабочий процесс

### 1. Подготовка данных
```bash
# 1. Поместите видео в Videos/
cp your_videos/*.mp4 Videos/

# 2. Извлеките кадры
python extract_frames.py
```

### 2. Обучение модели
```bash
# 3. Загрузите кадры в Roboflow (через веб-интерфейс)
# 4. Разметьте данные в Roboflow
# 5. Обучите модель в Roboflow
# 6. Скачайте обученную модель
```

### 3. Детекция
```bash
# 7. Используйте модель для детекции
python detect.py --source new_image.jpg
```

## 🌐 Работа на разных компьютерах

### 🔧 Настройка Git
```bash
# Клонирование проекта
git clone https://github.com/EmilioSedici16/TST_git_repository.git
cd TST_git_repository

# Установка зависимостей
pip install -r requirements.txt
```

### 🔄 Ежедневная работа
```bash
# Перед работой
git pull origin main

# После работы
git add .
git commit -m "Описание изменений"
git push origin main
```

### 📁 Синхронизация файлов
- ✅ **Код и конфигурация** - через Git
- ❌ **Видеофайлы и модели** - через облачное хранилище
- ✅ **Документация** - через Git

## 📦 Требования

### 🐍 Python пакеты
```
opencv-python>=4.5.0
roboflow>=1.0.0
ultralytics>=8.0.0
streamlit>=1.28.0
pillow>=8.0.0
numpy>=1.21.0
```

### 🖥️ Системные требования
- Python 3.8+
- Git (для синхронизации)
- 4GB+ RAM (для YOLO моделей)
- CUDA (опционально, для GPU ускорения)

### 🌐 Внешние сервисы
- Аккаунт Roboflow (бесплатный план доступен)
- GitHub репозиторий (для синхронизации)

## 🔧 Расширенные настройки

### 🎬 Извлечение кадров
```python
# Настройки в extract_frames.py
FRAME_EXTRACTION_SETTINGS = {
    'num_frames': 100,           # Количество кадров
    'start_offset': 0.1,         # Пропуск начала (10%)
    'end_offset': 0.1,           # Пропуск конца (10%)
    'image_format': 'jpg',       # Формат изображений
    'image_quality': 95          # Качество JPEG (0-100)
}
```

### 🤖 Модели детекции
```python
# Настройки в detect.py
DETECTION_SETTINGS = {
    'confidence': 0.5,           # Порог уверенности
    'iou_threshold': 0.45,       # Порог IoU для NMS
    'max_detections': 1000,      # Максимум детекций
    'device': 'auto'             # 'cpu', 'cuda', 'auto'
}
```

## 📊 Мониторинг и логирование

```python
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tst_project.log'),
        logging.StreamHandler()
    ]
)
```

## 🚨 Решение проблем

### ❌ Ошибки установки
```bash
# Обновление pip
python -m pip install --upgrade pip

# Установка с --user если проблемы с правами
pip install --user -r requirements.txt

# Использование conda
conda install opencv roboflow ultralytics streamlit
```

### 🐛 Проблемы с моделями
```bash
# Очистка кэша YOLO
rm -rf ~/.cache/ultralytics/

# Переустановка ultralytics
pip uninstall ultralytics
pip install ultralytics
```

### 🔑 Проблемы с Roboflow API
```bash
# Проверка API ключа
python -c "import os; print(os.environ.get('ROBOFLOW_API_KEY'))"

# Тест подключения
python -c "from roboflow import Roboflow; rf = Roboflow(); print('OK')"
```

## 📈 Производительность

### ⚡ Оптимизация извлечения кадров
- Используйте SSD для временных файлов
- Настройте количество кадров под задачу
- Используйте многопоточность для пакетной обработки

### 🚀 Ускорение детекции
- Используйте GPU (CUDA) для YOLO
- Уменьшите размер изображений
- Используйте более легкие модели (YOLOv8n vs YOLOv8x)

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit изменения (`git commit -m 'Add some AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Создайте Pull Request

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл `LICENSE` для подробностей.

## 📞 Поддержка

- 🐛 **Баги и предложения:** [GitHub Issues](https://github.com/EmilioSedici16/TST_git_repository/issues)
- 📚 **Документация:** См. файлы в папке документации
- 💬 **Вопросы:** Создайте Discussion в GitHub

---

**🎯 Цель проекта:** Создание полного pipeline для компьютерного зрения - от подготовки данных до детекции объектов с использованием современных инструментов и платформ.