# 🚀 Инструкция по настройке Git для проекта TST_Project

## Предварительные требования

1. **Убедитесь, что Git установлен:**
   - Скачайте Git с https://git-scm.com/download/win
   - Установите с настройками по умолчанию
   - Перезапустите терминал/Cursor после установки

2. **Настройте Git (если ещё не делали):**
   ```bash
   git config --global user.name "Ваше Имя"
   git config --global user.email "your.email@example.com"
   ```

## Шаги настройки репозитория

### 1. Инициализация Git репозитория
Откройте терминал в папке проекта и выполните:
```bash
git init
```

### 2. Добавление файлов в репозиторий
```bash
git add .
```

### 3. Создание первого коммита
```bash
git commit -m "Первый коммит: добавлены основные файлы проекта"
```

### 4. Создание репозитория на GitHub

1. Перейдите на https://github.com
2. Нажмите "New repository"
3. Назовите репозиторий: `TST_Project`
4. НЕ добавляйте README, .gitignore или лицензию (они уже есть)
5. Нажмите "Create repository"

### 5. Подключение к удалённому репозиторию
Замените `ВАШ_USERNAME` на ваш GitHub логин:
```bash
git remote add origin https://github.com/ВАШ_USERNAME/TST_Project.git
git branch -M main
git push -u origin main
```

## Ежедневная работа с Git

### Перед началом работы (на любом компьютере):
```bash
git pull origin main
```

### После завершения работы:
```bash
git add .
git commit -m "Описание внесённых изменений"
git push origin main
```

## Работа на разных компьютерах

### На новом компьютере:
1. Установите Git
2. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/ВАШ_USERNAME/TST_Project.git
   cd TST_Project
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

### Синхронизация изменений:
- **ВСЕГДА** делайте `git pull` перед началом работы
- **ВСЕГДА** делайте `git push` после завершения работы
- Избегайте одновременной работы на разных компьютерах

## Что исключено из репозитория (.gitignore)

✅ **Включено в Git:**
- Исходный код Python (*.py)
- Файлы настроек и документации
- requirements.txt
- Структура папок (Videos/, Images/)

❌ **Исключено из Git:**
- Видеофайлы (*.mp4, *.avi, etc.)
- Временные файлы Python (__pycache__)
- Сгенерированные изображения
- Настройки IDE

## Для больших видеофайлов

Рекомендуется хранить видеофайлы отдельно:
1. **Google Drive / OneDrive / Dropbox** - для синхронизации между компьютерами
2. **Git LFS** - для версионирования больших файлов (при необходимости)

## Полезные команды Git

```bash
git status              # Посмотреть статус изменений
git log --oneline       # История коммитов
git diff               # Посмотреть изменения
git reset --hard HEAD  # Отменить все локальные изменения (ОСТОРОЖНО!)
```

## Решение проблем

### Конфликты при pull:
```bash
git stash              # Сохранить локальные изменения
git pull origin main   # Получить изменения
git stash pop          # Восстановить локальные изменения
```

### Забыли сделать pull перед изменениями:
```bash
git stash
git pull origin main
git stash pop
# Разрешите конфликты если есть
git add .
git commit -m "Исправлены конфликты"
git push origin main
```

---

**📞 Нужна помощь?** Создайте issue в репозитории или обратитесь к документации Git: https://git-scm.com/doc