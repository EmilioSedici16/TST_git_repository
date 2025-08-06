# 📦 Инструкция для создания коммита

## 🚀 Быстрый способ (рекомендуется)

### Вариант 1: PowerShell скрипт
```powershell
# Правой кнопкой на Commit-Cleanup.ps1 → "Выполнить с помощью PowerShell"
# Или открыть PowerShell и выполнить:
.\Commit-Cleanup.ps1
```

### Вариант 2: Батник
```cmd
# Двойной клик на git_commit_cleanup.bat
# Или в cmd:
git_commit_cleanup.bat
```

## 🔧 Ручной способ

Если автоматические скрипты не работают, выполните в cmd или PowerShell:

```bash
# 1. Добавить все изменения
git add .

# 2. Создать коммит
git commit -m "🧹 Уборка рабочего стола: удалены устаревшие файлы, обновлена документация"

# 3. Отправить в GitHub
git push origin main
```

## 📋 Описание изменений для коммита

**Удалено 11 устаревших файлов:**
- README_extract_frames.md, ФИНАЛЬНАЯ_ИНСТРУКЦИЯ.md, ЗАПУСК_СЕЙЧАС.md
- ИНСТРУКЦИЯ_ЗАПУСКА.md, simple_extract.py
- run_extract.bat, run_extract.ps1, run_simple.bat
- auto_install_and_run.bat, find_and_run_python.bat
- install_and_run.ps1, promt.xml

**Обновлена документация:**
- README.md: актуализирована структура проекта
- docs/README_ROBOFLOW_PIPELINE.md: создан индекс документации
- docs/last_chat_export_03-01-2025.md: экспорт истории чата

**Результат:** чистая структура проекта, все функции сохранены в улучшенных версиях

## ✅ После коммита

Проект будет готов к:
- Синхронизации на втором ПК
- Продолжению работы с чистой структурой
- Дальнейшей разработке Roboflow pipeline