# ☁️ Синхронизация с UTair Cloud

## 🎯 Описание
Видеофайлы размещены в облачном хранилище UTair для синхронизации между компьютерами:
**📁 https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos**

## 🗂️ Структура проекта с облаком

```
ЛОКАЛЬНО (синхронизируется через Git):
TST_Project/
├── 📄 Код и скрипты (в Git)
├── 📚 Документация (в Git)
├── Images/ (локально генерируются)
└── Videos/ → 🔗 ССЫЛКА НА ОБЛАКО

ОБЛАКО UTair:
/TST_project/Videos/
├── 📹 1 секция_ задняя правая_20250723_101500.mp4
├── 📹 1 секция_ задняя правая_20250723_102500.mp4
├── 📹 1 секция_ задняя средняя_20250723_101500.mp4
├── 📹 1 секция_ задняя средняя_20250723_102500.mp4
├── 📹 Моечная средняя_2_20250723_101500.mp4
├── 📹 Моечная средняя_2_20250723_102500.mp4
├── 📹 Склад_ зона самовыдачи-1_20250527_105059.mp4
└── 📹 Склад_ зона самовыдачи-2_20250527_105047.mp4
```

## 🚀 Настройка синхронизации

### 🖥️ На ПЕРВОМ ПК (текущем):

#### 1. Создать символическую ссылку на облако
```bash
# Удалить локальную папку Videos (если есть файлы - скопировать в облако)
rmdir Videos /s /q

# Создать ссылку на смонтированный диск облака
mklink /D Videos "Z:\TST_project\Videos"
# Где Z: - это буква диска UTair Cloud
```

#### 2. Если облако не смонтировано как диск:

**Вариант А: WebDAV подключение**
```bash
# В Windows Explorer → Этот компьютер → Подключить сетевой диск
# Адрес: https://cloud.utair.ru/remote.php/dav/files/ВАШ_ЛОГИН/TST_project/Videos
```

**Вариант Б: Клиент синхронизации**
```bash
# Установить клиент UTair Cloud (если доступен)
# Настроить синхронизацию папки TST_project/Videos
```

**Вариант В: Ручная синхронизация**
```bash
# Скачивать/загружать файлы вручную через веб-интерфейс
```

### 💻 На ВТОРОМ ПК:

#### 1. Клонировать проект
```bash
git clone https://github.com/EmilioSedici16/TST_git_repository.git
cd TST_git_repository
pip install -r requirements.txt
```

#### 2. Настроить доступ к облаку
```bash
# Подключить UTair Cloud как сетевой диск
# И создать ссылку
mklink /D Videos "Z:\TST_project\Videos"
```

## 🔄 Рабочий процесс с облаком

### 📤 Добавление новых видео:

#### На любом ПК:
1. **Загрузить видео в облако:**
   - Перейти на https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos
   - Загрузить новые видеофайлы

2. **Обновить локальный проект:**
   ```bash
   # Видео автоматически появятся через синхронизацию
   # Или обновить ссылку/скачать вручную
   ```

3. **Извлечь кадры:**
   ```bash
   python extract_frames.py
   # Новые видео будут обработаны автоматически
   ```

### 🔄 Ежедневная работа:

#### Перед работой:
```bash
# 1. Обновить код
git pull origin main

# 2. Синхронизировать видео (автоматически или вручную)
# Проверить наличие новых файлов в Videos/
```

#### После работы:
```bash
# 1. Сохранить изменения кода
git add .
git commit -m "Обработка видео + новые результаты"
git push origin main

# 2. Загрузить новые видео в облако (если есть)
# Через веб-интерфейс или клиент синхронизации
```

## ⚙️ Автоматизация синхронизации

### 🔧 Создание скрипта синхронизации:

```batch
@echo off
echo 🔄 Синхронизация с UTair Cloud...

REM Проверка подключения к облаку
if exist Videos\ (
    echo ✅ Папка Videos доступна
) else (
    echo ❌ Папка Videos недоступна
    echo 🔧 Проверьте подключение к UTair Cloud
    pause
    exit /b 1
)

REM Подсчет видеофайлов
set /a count=0
for %%f in (Videos\*.mp4 Videos\*.avi Videos\*.mov) do set /a count+=1

echo 📊 Найдено видеофайлов: %count%

if %count% GTR 0 (
    echo ✅ Видео готовы к обработке
    set /p extract="Извлечь кадры из всех видео? (y/n): "
    if /i "%extract%"=="y" (
        python extract_frames.py
    )
) else (
    echo ⚠️ Видеофайлы не найдены
    echo 📥 Загрузите видео в облако: https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos
)

pause
```

### 🐍 Python скрипт для проверки облака:

```python
import os
import requests
from pathlib import Path

def check_cloud_sync():
    """Проверить синхронизацию с облаком"""
    videos_dir = Path("Videos")
    
    if not videos_dir.exists():
        print("❌ Папка Videos не найдена")
        print("🔧 Настройте подключение к UTair Cloud")
        return False
    
    # Подсчет локальных видео
    video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))
    
    print(f"📊 Локальных видеофайлов: {len(video_files)}")
    
    if len(video_files) == 0:
        print("⚠️ Видеофайлы не найдены локально")
        print("🔄 Проверьте синхронизацию с облаком")
        return False
    
    # Список файлов
    print("📹 Доступные видео:")
    for video in video_files:
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"   {video.name} ({size_mb:.1f} MB)")
    
    return True

if __name__ == "__main__":
    check_cloud_sync()
```

## 🔧 Настройка клиентов синхронизации

### 🌐 WebDAV (рекомендуемый):

#### Windows:
```
1. Этот компьютер → Подключить сетевой диск
2. Адрес: https://cloud.utair.ru/remote.php/dav/files/ВАШ_ЛОГИН/
3. Выбрать букву диска (например, Z:)
4. Указать логин/пароль UTair
5. Создать ссылку: mklink /D Videos "Z:\TST_project\Videos"
```

#### Альтернативные клиенты:
- **WinSCP** (SFTP/WebDAV)
- **Cyberduck** (WebDAV)
- **rclone** (командная строка)

### 📱 Мобильный доступ:
- Приложение UTair Cloud (если доступно)
- Мобильный браузер: https://cloud.utair.ru

## 🚨 Решение проблем

### ❌ Папка Videos недоступна:
```bash
# Проверить подключение к сети
ping cloud.utair.ru

# Переподключить сетевой диск
net use Z: /delete
net use Z: https://cloud.utair.ru/remote.php/dav/files/ВАШ_ЛОГИН/
```

### 🔄 Медленная синхронизация:
```bash
# Использовать только для новых файлов
# Настроить исключения в .gitignore
# Использовать сжатие видео для ускорения загрузки
```

### 🔐 Проблемы с доступом:
```bash
# Проверить права доступа в UTair Cloud
# Убедиться, что папка TST_project/Videos существует
# Проверить логин/пароль
```

## 📊 Мониторинг синхронизации

### 📈 Статистика использования:
```python
# Скрипт для мониторинга
import os
from pathlib import Path

videos_dir = Path("Videos")
if videos_dir.exists():
    video_files = list(videos_dir.glob("*.mp4"))
    total_size = sum(f.stat().st_size for f in video_files)
    
    print(f"📊 Статистика Videos:")
    print(f"   Файлов: {len(video_files)}")
    print(f"   Размер: {total_size / (1024**3):.2f} GB")
    print(f"   Среднее: {total_size / len(video_files) / (1024**2):.1f} MB/файл")
```

### 🔍 Проверка целостности:
```bash
# Проверить, что все файлы доступны
python -c "
import os
videos = [f for f in os.listdir('Videos') if f.endswith('.mp4')]
print(f'Доступно видео: {len(videos)}')
for v in videos:
    try:
        size = os.path.getsize(f'Videos/{v}')
        print(f'✅ {v}: {size/1024/1024:.1f} MB')
    except:
        print(f'❌ {v}: Недоступен')
"
```

## 🎯 Рекомендации

### ✅ Лучшие практики:
1. **Регулярная синхронизация** - проверяйте облако перед работой
2. **Осмысленные имена файлов** - сохраняйте текущую схему именования
3. **Резервное копирование** - дублируйте важные видео
4. **Сжатие** - используйте сжатие для экономии места и ускорения

### 📁 Организация файлов:
```
UTair Cloud: /TST_project/
├── Videos/                    # Исходные видео
│   ├── Секция1/              # Группировка по зонам
│   ├── Моечная/
│   └── Склад/
├── Archive/                   # Архив старых видео
└── Processed/                 # Обработанные данные (если нужно)
```

### 🔄 Автоматизация:
```bash
# Создать задачу в планировщике Windows
# Ежедневная проверка синхронизации
schtasks /create /tn "UTair_Cloud_Sync" /tr "python check_cloud_sync.py" /sc daily
```

## 📞 Поддержка

- **UTair Cloud:** Техподдержка UTair
- **Проект:** https://github.com/EmilioSedici16/TST_git_repository/issues
- **WebDAV помощь:** https://docs.nextcloud.com/desktop/latest/navigating.html

---

**🌐 Ваше облако:** https://cloud.utair.ru/apps/files/?dir=/TST_project/Videos
**📁 Локальная ссылка:** `Videos/` → облако
**🔄 Синхронизация:** Автоматическая через WebDAV или ручная через веб-интерфейс