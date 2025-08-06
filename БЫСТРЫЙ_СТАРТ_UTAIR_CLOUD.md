# ⚡ Быстрый старт с UTair Cloud

## 🌐 Ваши видео в облаке
**📁 https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos**

## 🚀 Настройка на первом ПК

### 1️⃣ Объединить с Roboflow проектом
```
Дважды кликните: ОБЪЕДИНИТЬ_С_ROBOFLOW.bat
Выберите: Вариант 2 (объединить)
```

### 2️⃣ Настроить синхронизацию с облаком
```
Дважды кликните: НАСТРОИТЬ_ОБЛАКО.bat
Выберите: Вариант 1 (WebDAV)
```

### 3️⃣ Проверить и обработать видео
```bash
# Проверка синхронизации
python check_cloud_sync.py --interactive

# Извлечение кадров
python extract_frames.py
```

## 💻 Настройка на втором ПК

### 1️⃣ Клонировать проект
```bash
git clone https://github.com/EmilioSedici16/TST_git_repository.git
cd TST_git_repository
pip install -r requirements.txt
```

### 2️⃣ Настроить UTair Cloud
```
Дважды кликните: НАСТРОИТЬ_ОБЛАКО.bat
```

### 3️⃣ Проверить доступ к видео
```bash
python check_cloud_sync.py
```

## 🔄 Ежедневная работа

### ▶️ Перед работой (любой ПК):
```bash
git pull origin main
python check_cloud_sync.py --quick
```

### 🎬 Обработка видео:
```bash
python extract_frames.py
python ПОЛНЫЙ_PIPELINE.py --prepare-roboflow
```

### ◀️ После работы:
```bash
git add .
git commit -m "Обновления проекта"
git push origin main
```

## 🛠️ Полезные команды

```bash
# Проверка видео в облаке
python check_cloud_sync.py

# Интерактивный режим
python check_cloud_sync.py --interactive

# Полный pipeline
python ПОЛНЫЙ_PIPELINE.py --full

# Веб-интерфейс
streamlit run app.py
```

## 🚨 Решение проблем

### ❌ Videos недоступна:
```
НАСТРОИТЬ_ОБЛАКО.bat → Вариант 1 (WebDAV)
```

### 🔄 Медленная синхронизация:
- Используйте WebDAV подключение
- Проверьте интернет-соединение

### 📁 Новые видео не видны:
```bash
# Проверить облако в браузере
https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos

# Обновить локальную ссылку
python check_cloud_sync.py
```

## 📊 Контроль синхронизации

### 📈 Статистика:
```bash
python check_cloud_sync.py --quick
```

### 🔍 Подробная проверка:
```bash
python check_cloud_sync.py
```

---

**🌐 Облако:** https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos  
**💻 GitHub:** https://github.com/EmilioSedici16/TST_git_repository  
**📚 Документация:** `СИНХРОНИЗАЦИЯ_UTAIR_CLOUD.md`