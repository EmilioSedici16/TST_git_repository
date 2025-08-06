# 🎯 ФИНАЛЬНАЯ ИНСТРУКЦИЯ: Работа с существующим GitHub репозиторием

## 📍 Ваша текущая ситуация

✅ **У вас есть:**
- Локальный проект `TST_Project` (извлечение кадров из видео)
- GitHub репозиторий: https://github.com/EmilioSedici16/TST_git_repository (проект компьютерного зрения)

🎯 **Цель:** Синхронизировать проекты для работы на двух компьютерах

## 🚀 ПОШАГОВЫЙ ПЛАН

### ШАГ 1: На текущем ПК (подключение к GitHub)

#### 🎬 Самый простой способ:
```
Дважды кликните на файл: connect_to_github.bat
```

Скрипт автоматически:
- Проверит Git
- Подключит к вашему GitHub репозиторию
- Предложит варианты синхронизации

#### 🔧 Ручной способ:
```bash
# 1. Инициализация (если нужно)
git init
git add .
git commit -m "Локальные файлы TST_Project"

# 2. Подключение к GitHub
git remote add origin https://github.com/EmilioSedici16/TST_git_repository.git

# 3. Выберите один из вариантов:

# ВАРИАНТ A: Заменить содержимое GitHub на ваши файлы
git push -u origin main --force

# ВАРИАНТ B: Объединить с файлами из GitHub
git pull origin main --allow-unrelated-histories
git push origin main
```

### ШАГ 2: На втором ПК (получение проекта)

#### 1. Установить программы:
- Git: https://git-scm.com/download/win
- Python: https://python.org/downloads
- Cursor: https://cursor.sh

#### 2. Настроить Git:
```bash
git config --global user.name "EmilioSedici16"
git config --global user.email "ваш_email@example.com"
```

#### 3. Клонировать проект:
```bash
cd Documents
git clone https://github.com/EmilioSedici16/TST_git_repository.git
cd TST_git_repository

# Переименовать папку (по желанию)
cd ..
ren TST_git_repository TST_Project
cd TST_Project

# Установить зависимости
pip install -r requirements.txt
```

### ШАГ 3: Ежедневная работа (на любом ПК)

#### ▶️ ПЕРЕД началом работы:
```bash
cd TST_Project  # или TST_git_repository
git pull origin main
```

#### ◀️ ПОСЛЕ завершения работы:
```bash
git add .
git commit -m "Описание изменений"
git push origin main
```

## 📁 Что происходит с файлами

### 🔄 Процесс объединения:

**Если выберете "объединить файлы":**
- Ваши файлы TST_Project (извлечение кадров)
- + Файлы из GitHub (компьютерное зрение)
- = Объединённый проект с обеими функциями

**Если выберете "заменить содержимое GitHub":**
- Только ваши файлы TST_Project
- GitHub получит ваши файлы извлечения кадров

### 📋 Структура после объединения:
```
TST_Project/
├── 📹 Ваши файлы (извлечение кадров):
│   ├── extract_frames.py
│   ├── simple_extract.py
│   ├── requirements.txt
│   ├── Videos/
│   ├── Images/
│   └── run_extract.bat
│
├── 🤖 Файлы из GitHub (компьютерное зрение):
│   ├── app.py
│   ├── detect.py
│   ├── safety_detection.py
│   ├── yolov8n.pt
│   └── data/
│
└── 🔧 Git файлы:
    ├── .gitignore
    ├── README.md
    └── инструкции...
```

## ⚠️ Важные моменты

### 🚫 Что НЕ синхронизируется:
- Видеофайлы (*.mp4) - исключены в .gitignore
- Сгенерированные изображения
- Временные файлы Python

### 💡 Для видеофайлов используйте:
- Google Drive / OneDrive / Dropbox
- Или копирование через USB

### 🔐 Аутентификация:
При первом push может потребоваться:
- Логин: `EmilioSedici16`
- Пароль: Personal Access Token (не обычный пароль)

Создать токен: GitHub → Settings → Developer settings → Personal access tokens

## 🔧 Возможные проблемы и решения

### ❌ "fatal: remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/EmilioSedici16/TST_git_repository.git
```

### ❌ Ошибка аутентификации
```bash
# Используйте Personal Access Token вместо пароля
# Создайте токен на GitHub и используйте его как пароль
```

### ⚠️ Конфликты файлов
```bash
git status  # Посмотреть конфликты
# Открыть файлы с конфликтами и разрешить их
git add .
git commit -m "Разрешены конфликты"
git push origin main
```

### 🔍 Проверка статуса
```bash
git status          # Что изменилось
git log --oneline   # История изменений
git remote -v       # Подключённые репозитории
```

## 🎉 Результат

После выполнения всех шагов:
- ✅ Локальный проект подключён к GitHub
- ✅ Синхронизация между компьютерами работает
- ✅ История изменений сохраняется
- ✅ Можно работать на любом компьютере

## 📞 Поддержка

### 📚 Файлы-справочники:
- `ПОДКЛЮЧЕНИЕ_К_СУЩЕСТВУЮЩЕМУ_REPO.md` - подробная инструкция
- `РАБОТА_НА_ВТОРОМ_ПК.md` - настройка второго компьютера
- `ШПАРГАЛКА_ВТОРОЙ_ПК.md` - быстрый справочник

### 🆘 Если что-то не работает:
1. Прочитайте полную инструкцию в `ПОДКЛЮЧЕНИЕ_К_СУЩЕСТВУЮЩЕМУ_REPO.md`
2. Создайте issue: https://github.com/EmilioSedici16/TST_git_repository/issues
3. Проверьте, что Git правильно настроен и у вас есть права на репозиторий

---

## 🎯 СЛЕДУЮЩИЙ ШАГ

**Запустите файл `connect_to_github.bat` и следуйте инструкциям!**

🌐 **Ваш репозиторий:** https://github.com/EmilioSedici16/TST_git_repository